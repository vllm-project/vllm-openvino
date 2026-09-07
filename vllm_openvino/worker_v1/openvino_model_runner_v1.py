# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Tuple

import numpy as np
import openvino as ov
import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.multimodal import BatchedTensorInputs
from vllm.sampling_params import SamplingType
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

from vllm_openvino.attention.backends.openvino import OpenVINOAttentionMetadata
from vllm_openvino.model_executor.model_loader.openvino import get_model
from vllm_openvino.types import ModelInput

logger = init_logger(__name__)


class OpenVINOModelRunnerV1:
    def __init__(
        self,
        ov_core: ov.Core,
        vllm_config: VllmConfig,
        kv_cache_dtype: Optional[str] = "auto",
    ):
        self.ov_core = ov_core
        self.vllm_config = vllm_config
        self.device = vllm_config.device_config.device
        self.kv_cache_dtype = kv_cache_dtype
        self.block_size = vllm_config.cache_config.block_size
        self.model: nn.Module  # Set after load_model()

        self.requests: dict[str, CachedRequestState] = {}

        max_model_len = vllm_config.model_config.max_model_len
        block_size = vllm_config.cache_config.block_size
        self.input_batch = InputBatch(
            max_num_reqs=self.vllm_config.scheduler_config.max_num_seqs,
            max_model_len=max_model_len,
            max_num_batched_tokens=(
                vllm_config.scheduler_config.max_num_batched_tokens),
            device=self.device,
            vocab_size=vllm_config.model_config.get_vocab_size(),
            block_sizes=[block_size],
            kernel_block_sizes=[block_size],
            max_num_blocks_per_req=[cdiv(max_model_len, block_size)],
        )

    def load_model(self) -> None:
        self.model = get_model(vllm_config=self.vllm_config,
                               kv_cache_dtype=self.kv_cache_dtype,
                               ov_core=self.ov_core)

    def get_model(self) -> nn.Module:
        return self.model

    @torch.inference_mode()
    def warm_up_sampler(self) -> None:
        num_reqs = min(
            self.vllm_config.scheduler_config.max_num_seqs,
            self.vllm_config.scheduler_config.max_num_batched_tokens,
        )
        vocab_size = self.vllm_config.model_config.get_vocab_size()
        dummy_tensors = lambda value: torch.full(
            (num_reqs,), value, device=self.device)
        sampling_metadata = SamplingMetadata(
            temperature=dummy_tensors(0.5),
            all_greedy=False,
            all_random=False,
            top_p=dummy_tensors(0.9),
            top_k=dummy_tensors(vocab_size - 1),
            generators={},
            max_num_logprobs=None,
            logprob_token_ids=None,
            no_penalties=True,
            prompt_token_ids=None,
            frequency_penalties=dummy_tensors(0.1),
            presence_penalties=dummy_tensors(0.1),
            repetition_penalties=dummy_tensors(0.1),
            output_token_ids=[[] for _ in range(num_reqs)],
            spec_token_ids=[[] for _ in range(num_reqs)],
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=LogitsProcessors(),
        )

        # vLLM's CPU sampler is torch-compiled lazily. Run it twice so both
        # compilation and one-time CPU kernel setup finish before serving.
        for _ in range(2):
            logits = torch.rand((num_reqs, vocab_size), device=self.device)
            self.model.sample(logits, sampling_metadata)

    def _update_states(self, scheduler_output: SchedulerOutput) -> None:
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            self.input_batch.remove_request(req_id)

        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        # Remove unscheduled requests from the persistent batch but retain
        # their cached state. Remove resumed requests as well so stale block
        # tables are cleared before they are re-added.
        for req_id in cached_req_ids - (scheduled_req_ids - resumed_req_ids):
            self.input_batch.remove_request(req_id)

        reqs_to_add: list[CachedRequestState] = []
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            sampling_params = new_req_data.sampling_params
            if (sampling_params is not None and
                    sampling_params.sampling_type == SamplingType.RANDOM_SEED):
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            req_state = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                prompt_embeds=new_req_data.prompt_embeds,
                prompt_is_token_ids=new_req_data.prompt_is_token_ids,
                mm_features=new_req_data.mm_features,
                sampling_params=sampling_params,
                pooling_params=new_req_data.pooling_params,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )
            self.requests[req_id] = req_state
            reqs_to_add.append(req_state)

        req_data = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            req_state.num_computed_tokens = num_computed_tokens
            num_output_tokens = req_data.num_output_tokens[i]
            # Align local output history if the scheduler rolled tokens back.
            if num_output_tokens < len(req_state.output_token_ids):
                del req_state.output_token_ids[num_output_tokens:]

            new_block_ids = req_data.new_block_ids[i]
            # Resumed requests replace their block tables; running requests
            # append only blocks allocated since the previous step.
            if req_id in req_data.resumed_req_ids:
                assert new_block_ids is not None
                req_state.block_ids = new_block_ids
            elif new_block_ids is not None:
                for block_ids, new_ids in zip(req_state.block_ids,
                                              new_block_ids):
                    block_ids.extend(new_ids)

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                reqs_to_add.append(req_state)
                continue

            self.input_batch.num_computed_tokens_cpu[req_index] = (
                num_computed_tokens)
            self.input_batch.num_tokens_no_spec[req_index] = req_state.num_tokens
            if new_block_ids is not None:
                self.input_batch.block_table.append_row(new_block_ids, req_index)

        for req_state in reqs_to_add:
            self.input_batch.add_request(req_state)

        self.input_batch.condense()
        self.input_batch.refresh_metadata()

    def _prepare_model_input(self, scheduler_output) -> ModelInput:
        """Prepare the model input based on scheduled requests.
        """
        input_tokens = []
        input_positions = []
        seq_lens = []
        past_lens = []
        query_lens = []

        subsequence_begins = []
        block_indices = []
        block_indices_begins = []

        subsequence_begins.append(0)
        block_indices_begins.append(0)

        for req_id in self.input_batch.req_ids:
            request = self.requests[req_id]
            if len(request.block_ids) != 1:
                raise NotImplementedError(
                    "OpenVINO currently supports one KV cache group")
            block_table = request.block_ids[0]

            block_indices.extend(block_table)
            block_indices_begins.append(block_indices_begins[-1] +
                                        len(block_table))
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens.get(
                req_id, 0)
            last_token_position = num_scheduled_tokens + request.num_computed_tokens
            # Materialize this step's scheduled suffix, crossing from prompt
            # tokens into generated tokens when necessary.
            tokens = [] if request.num_computed_tokens >= len(request.prompt_token_ids) else request.prompt_token_ids[request.num_computed_tokens:last_token_position]
            tokens += request.output_token_ids[request.num_computed_tokens - len(request.prompt_token_ids): last_token_position - len(request.prompt_token_ids)]
            seq_len = len(tokens) + request.num_computed_tokens
            seq_lens.append(seq_len)
            query_len = len(tokens)
            query_lens.append(query_len)
            input_tokens.extend(tokens)
            positions_range = range(request.num_computed_tokens, seq_len)
            input_positions.extend(list(positions_range))

            past_lens.append(request.num_computed_tokens)
            subsequence_begins.append(subsequence_begins[-1] + query_len)

        # Each packed-sequence endpoint identifies the token whose logits are
        # gathered for sampling.
        sampled_token_indices = np.array(subsequence_begins[1:]) - 1

        input_tokens = ov.Tensor(np.array(input_tokens, dtype=np.int64))

        input_positions = ov.Tensor(np.array(input_positions, dtype=np.int64))
        sampled_token_indices_tensor = ov.Tensor(np.array(sampled_token_indices, dtype=np.int64))

        past_lens_tensor = ov.Tensor(np.array(past_lens, dtype=np.int32))
        subsequence_begins_tensor = ov.Tensor(np.array(subsequence_begins, dtype=np.int32))
        block_indices_tensor = ov.Tensor(np.array(block_indices, dtype=np.int32))
        block_indices_begins_tensor = ov.Tensor(np.array(block_indices_begins, dtype=np.int32))
        max_context_len_tensor = ov.Tensor(np.array(max(seq_lens), dtype=np.int32))

        attn_metadata = OpenVINOAttentionMetadata(
            past_lens=past_lens_tensor,
            subsequence_begins=subsequence_begins_tensor,
            block_indices=block_indices_tensor,
            block_indices_begins=block_indices_begins_tensor,
            max_context_len=max_context_len_tensor,
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=False,
            sampled_token_indices=sampled_token_indices_tensor
        )

        return ModelInput(
            input_tokens,
            input_positions,
            attn_metadata,
            seq_lens,
            query_lens,
            multi_modal_kwargs=None,
        )

    def prepare_input_tensors(
        self,
        scheduler_output
    ) -> Tuple[torch.Tensor, torch.Tensor, OpenVINOAttentionMetadata,
               SamplingMetadata, BatchedTensorInputs]:
        # Prepare input tensors.
        (
            input_tokens,
            input_positions,
            attn_metadata,
            seq_lens,
            query_lens,
            multi_modal_kwargs,
        ) = self._prepare_model_input(scheduler_output)

        sampling_metadata = self.input_batch.sampling_metadata

        return (
            input_tokens,
            input_positions,
            attn_metadata,
            sampling_metadata,
            multi_modal_kwargs,
        )

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output,
        kv_caches: List[Tuple["ov.Tensor", "ov.Tensor"]],
    ) -> ModelRunnerOutput:
        self._update_states(scheduler_output)

        (
            input_tokens,
            input_positions,
            attn_metadata,
            sampling_metadata,
            multi_modal_kwargs,
        ) = self.prepare_input_tensors(scheduler_output)

        model_executable = self.model
        execute_model_kwargs = {
            "input_ids":
            input_tokens,
            "positions":
            input_positions,
            "kv_caches":
            kv_caches,
        }

        with set_forward_context(attn_metadata, self.vllm_config, 0):
            hidden_states = model_executable(**execute_model_kwargs)

        logits = self.model.compute_logits(hidden_states, None)

        # Sample the next token and get logprobs if needed.
        sampling_metadata = self.input_batch.sampling_metadata

        sampler_output = self.model.sample(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )

        sampled_tokens = sampler_output.sampled_token_ids.tolist()

        logprobs_lists = sampler_output.logprobs_tensors.tolists() \
            if sampler_output.logprobs_tensors is not None else None

        valid_sampled_tokens = sampled_tokens

        for i, req_id in enumerate(self.input_batch.req_ids):
            req_state = self.requests[req_id]
            seq_len = (req_state.num_computed_tokens +
                        scheduler_output.num_scheduled_tokens.get(req_id, 0))
            # Ignore the sampled token for partial prefills.
            if seq_len < req_state.num_tokens:
                valid_sampled_tokens[i] = []

            sampled_ids = valid_sampled_tokens[i]
            if sampled_ids:
                # Single-rank V1 does not return sampled IDs in the next
                # SchedulerOutput, so cache them locally for the next step.
                req_index = self.input_batch.req_id_to_index[req_id]
                start_idx = self.input_batch.num_tokens_no_spec[req_index]
                end_idx = start_idx + len(sampled_ids)
                self.input_batch.token_ids_cpu[
                    req_index, start_idx:end_idx] = sampled_ids
                self.input_batch.is_token_ids[
                    req_index, start_idx:end_idx] = True
                self.input_batch.num_tokens_no_spec[req_index] = end_idx
                req_state.output_token_ids.extend(sampled_ids)

        return ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=valid_sampled_tokens,
            logprobs=logprobs_lists,
            prompt_logprobs_dict={},
        )

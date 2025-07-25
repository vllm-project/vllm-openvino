# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Tuple

import numpy as np
import openvino as ov
import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.multimodal import (BatchedTensorInputs,
                             MultiModalKwargs)
from vllm.sampling_params import SamplingType
from vllm.utils import (cdiv,
                        is_pin_memory_available)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

from vllm_openvino.attention.backends.openvino import OpenVINOAttentionMetadata
from vllm_openvino.model_executor.model_loader.openvino import get_model
from vllm_openvino.worker.openvino_model_runner import ModelInput

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
        self.model: nn.Module  # Set after load_model()

        self.requests: dict[str, CachedRequestState] = {}

        self.input_batch = InputBatch(
            max_num_reqs=self.vllm_config.scheduler_config.max_num_seqs,
            max_model_len=vllm_config.model_config.max_model_len,
            max_num_blocks_per_req=cdiv(vllm_config.model_config.max_model_len, vllm_config.cache_config.block_size),
            device=self.device,
            pin_memory=is_pin_memory_available(),
            vocab_size=vllm_config.model_config.get_vocab_size(),
        )

    def load_model(self) -> None:
        self.model = get_model(vllm_config=self.vllm_config,
                               kv_cache_dtype=self.kv_cache_dtype,
                               ov_core=self.ov_core)

    def get_model(self) -> nn.Module:
        return self.model

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        removed_req_indices: list[int] = []

        # Remove finished requests from the batch
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            req_index = self.input_batch.remove_request(req_id)
            if req_index is not None:
                removed_req_indices.append(req_index)

        # Remove the unscheduled requests from the batch
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        unscheduled_req_ids = cached_req_ids - scheduled_req_ids
        for req_id in unscheduled_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            assert req_index is not None
            removed_req_indices.append(req_index)

        req_ids_to_add: list[str] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            sampling_params = new_req_data.sampling_params
            if sampling_params.sampling_type == SamplingType.RANDOM_SEED:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                prompt=new_req_data.prompt,
                mm_inputs=new_req_data.mm_inputs,
                mm_positions=new_req_data.mm_positions,
                sampling_params=sampling_params,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )
            req_ids_to_add.append(req_id)

        # Update the states of the running/resumed requests.
        for req_data in scheduler_output.scheduled_cached_reqs:
            req_id = req_data.req_id
            req_state = self.requests[req_id]

            # Update the requests states.
            num_computed_tokens = req_data.num_computed_tokens
            req_state.num_computed_tokens = num_computed_tokens
            # Add the sampled token(s) from the previous step (if any).
            num_new_tokens = (num_computed_tokens +
                              len(req_data.new_token_ids) -
                              req_state.num_tokens)
            if num_new_tokens == 1:
                req_state.output_token_ids.append(req_data.new_token_ids[-1])
            elif num_new_tokens > 0:
                req_state.output_token_ids.extend(
                    req_data.new_token_ids[-num_new_tokens:])
            # Update the block IDs.
            if not req_data.resumed_from_preemption:
                # Append the new blocks to the existing block IDs.
                req_state.block_ids.extend(req_data.new_block_ids)
            else:
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = req_data.new_block_ids

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                # The request is not in the batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.
                req_ids_to_add.append(req_id)
                continue

            # Update block tables and number of computed tokens
            self.input_batch.block_table.append_row(req_data.new_block_ids,
                                                    req_index)
            end_token_index = num_computed_tokens + len(req_data.new_token_ids)
            self.input_batch.num_tokens[req_index] = end_token_index

        batch_changed = len(removed_req_indices) > 0 or len(req_ids_to_add) > 0

        # Add the new or resumed requests to the batch.
        removed_req_indices = sorted(removed_req_indices, reverse=True)
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            if removed_req_indices:
                req_index = removed_req_indices.pop()
            else:
                req_index = None
            self.input_batch.add_request(req_state, req_index)

        # Process sampling params so they match V1 sampler format
        if removed_req_indices:
            self.input_batch.condense(removed_req_indices)

        if batch_changed:
            self.input_batch.refresh_sampling_metadata()

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

        if len(self.requests) == 0:
            return ModelInput.empty(self.device)

        for req_id in self.input_batch.req_ids:
            request = self.requests[req_id]
            block_table = request.block_ids

            block_indices.extend(block_table)
            block_indices_begins.append(block_indices_begins[-1] +
                                        len(block_table))
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            last_token_position = num_scheduled_tokens + request.num_computed_tokens
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

        sampled_token_indices = np.array(subsequence_begins[1:]) - 1

        max_query_len = max(query_lens)
        assert max_query_len > 0, "Invalid query_lens: {}".format(query_lens)

        input_tokens = ov.Tensor(np.array(input_tokens), ov.Shape([len(input_tokens)]), ov.Type.i64)

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
            **MultiModalKwargs.as_kwargs(multi_modal_kwargs or {},
                                         device=self.device),
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

        logprobs_lists = sampler_output.logprobs_tensors.tolist() \
            if sampler_output.logprobs_tensors is not None else None

        valid_sampled_tokens = sampled_tokens

        for i, req_id in enumerate(self.input_batch.req_ids):
            req_state = self.requests[req_id]
            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])
            # Ignore the sampled token for partial prefills.
            if seq_len < req_state.num_tokens:
                valid_sampled_tokens[i] = []

        return ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=valid_sampled_tokens,
            spec_token_ids=None,
            logprobs=logprobs_lists,
            prompt_logprobs_dict={},
        )

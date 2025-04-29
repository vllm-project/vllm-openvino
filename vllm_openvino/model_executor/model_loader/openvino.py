# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: SIM117
from pathlib import Path
from typing import Optional

import openvino as ov
import torch
from huggingface_hub import HfApi
from openvino._offline_transformations import paged_attention_transformation
from optimum.intel import OVModelForCausalLM
from torch import nn

import vllm_openvino.envs as envs
from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import (LogitsProcessor,
                                                         _prune_hidden_states)
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata

logger = init_logger(__name__)


def _flatten_inputs(inputs):
    """
    Helper function for making nested inputs flattens
    """
    flatten_inputs = []
    for input_data in inputs:
        if input_data is None:
            continue
        if isinstance(input_data, (list, tuple)):
            flatten_inputs.extend(_flatten_inputs(input_data))
        elif isinstance(input_data, dict):
            flatten_inputs.extend(_flatten_inputs(list(input_data.values())))
        else:
            flatten_inputs.append(input_data)
    return flatten_inputs


def _modify_cache_parameters(model: ov.Model, kv_cache_dtype: ov.Type):
    # set global KV cache precision if kv_cache_dtype is defined
    if kv_cache_dtype != ov.Type.dynamic:
        model.set_rt_info(kv_cache_dtype, ["runtime_options", "KV_CACHE_PRECISION"])

    for parameter in model.get_parameters():
        input = parameter.get_output_tensor(0)
        input_names = input.get_names()
        if len(input_names) != 1:
            continue
        input_name = next(iter(input_names))
        is_key_cache = input_name.startswith("key_cache.")
        is_value_cache = input_name.startswith("value_cache.")

        if is_key_cache or is_value_cache:
            shape = parameter.get_partial_shape()
            num_heads = shape[1].get_length()
            head_size = shape[2].get_length()
            # set parameters as dynamic to infer actual shape / type in plugin
            parameter.set_partial_shape(ov.PartialShape.dynamic(4))
            parameter.set_element_type(ov.Type.undefined)
            # specify actual KV cache parameters via runtime information of PagedAttention operation
            pa_op = next(iter(parameter.output(0).get_target_inputs())).get_node()
            pa_op.get_rt_info()["num_k_heads" if is_key_cache else "num_v_heads"] = num_heads
            pa_op.get_rt_info()["k_head_size" if is_key_cache else "v_head_size"] = head_size

    model.validate_nodes_and_infer_types()


def _require_model_export(model_id, revision=None, subfolder=None):
    model_dir = Path(model_id)
    if subfolder is not None:
        model_dir = model_dir / subfolder
    if model_dir.is_dir():
        return (not (model_dir / "openvino_model.xml").exists()
                or not (model_dir / "openvino_model.bin").exists())

    hf_api = HfApi()
    try:
        model_info = hf_api.model_info(model_id, revision=revision or "main")
        normalized_subfolder = (None if subfolder is None else
                                Path(subfolder).as_posix())
        model_files = [
            file.rfilename for file in model_info.siblings
            if normalized_subfolder is None
            or file.rfilename.startswith(normalized_subfolder)
        ]
        ov_model_path = ("openvino_model.xml" if normalized_subfolder is None
                         else f"{normalized_subfolder}/openvino_model.xml")
        return (ov_model_path not in model_files
                or ov_model_path.replace(".xml", ".bin") not in model_files)
    except Exception:
        return True


class OpenVINOCausalLM(nn.Module):

    def __init__(
        self,
        ov_core: ov.Core,
        model_config: ModelConfig,
        kv_cache_dtype: ov.Type,
    ) -> None:
        super().__init__()
        self.logits_processor = LogitsProcessor(
            model_config.hf_config.vocab_size, logits_as_input=True)
        self.sampler = Sampler()

        export = _require_model_export(model_config.model)
        if export:
            logger.warning(
                f"Provided model id {model_config.model} does not "  # noqa: G004
                "contain OpenVINO IR, the model will be converted to IR with "
                "default options. If you need to use specific options for "
                "model conversion, use optimum-cli export openvino with "
                "desired options.")
        else:
            logger.warning(
                "OpenVINO IR is available for provided model id "  # noqa: G004
                f"{model_config.model}. This IR will be used for inference "
                "as-is, all possible options that may affect model conversion "
                "are ignored.")

        load_in_8bit = (envs.VLLM_OPENVINO_ENABLE_QUANTIZED_WEIGHTS
                        if export else False)
        pt_model = OVModelForCausalLM.from_pretrained(
            model_config.model,
            export=export,
            compile=False,
            load_in_8bit=load_in_8bit,
            trust_remote_code=model_config.trust_remote_code,
        )

        # apply Paged Attention transformation
        paged_attention_transformation(pt_model.model)
        # then set dynamic shapes and precisions for KV cache, so plugins
        # will automatically resolve them during compile_model
        _modify_cache_parameters(pt_model.model, kv_cache_dtype)

        ov_device = envs.VLLM_OPENVINO_DEVICE
        ov_compiled = ov_core.compile_model(pt_model.model, ov_device)
        self.ov_request = ov_compiled.create_infer_request()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: list[tuple[ov.Tensor, ov.Tensor]],
    ) -> torch.Tensor:
        flat_kv_caches = _flatten_inputs(kv_caches)
        attn_metadata = get_forward_context().attn_metadata

        inputs = [
            input_ids,
            positions,
            *flat_kv_caches,
            attn_metadata.past_lens,
            attn_metadata.subsequence_begins,
            attn_metadata.block_indices,
            attn_metadata.block_indices_begins,
            attn_metadata.max_context_len,
        ]

        self.ov_request.start_async(inputs, share_inputs=True)
        self.ov_request.wait()

        logits = torch.from_numpy(self.ov_request.get_tensor("logits").data)

        # TODO: remove 'view' once OpenVINO PA will drop 'seq_len' dimension
        return logits.view(-1, logits.shape[-1])

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        hidden_states = _prune_hidden_states(hidden_states, sampling_metadata)
        logits = self.logits_processor(None, hidden_states, sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens


def get_model(
    vllm_config: VllmConfig,
    kv_cache_dtype: ov.Type,
    **kwargs,
) -> torch.nn.Module:
    lora_config = kwargs.get("lora_config")
    ov_core = kwargs.get("ov_core")
    if lora_config:
        raise ValueError(
            "OpenVINO modeling does not support LoRA, "
            "but LoRA is enabled. Support for this model may "
            "be added in the future. If this is important to you, "
            "please open an issue on github.")

    with set_current_vllm_config(vllm_config):
        return OpenVINOCausalLM(ov_core, vllm_config.model_config,
                                kv_cache_dtype)
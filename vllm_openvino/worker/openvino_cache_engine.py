# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import openvino as ov
from vllm.config import CacheConfig, DeviceConfig, ModelConfig, ParallelConfig
from vllm.platforms import current_platform

from vllm_openvino.attention.backends.openvino import OpenVINOAttentionBackend

str_to_ov_type = {
    "u8": ov.Type.u8,
    "i8": ov.Type.i8,
    "fp16": ov.Type.f16,
    "f16": ov.Type.f16,
    "bf16": ov.Type.bf16,
    "f32": ov.Type.f32,
    "fp32": ov.Type.f32,
    "dynamic": ov.Type.dynamic,
}


class OpenVINOCacheEngine:
    """Allocate OpenVINO KV-cache tensors and perform block operations."""

    def __init__(self, cache_config: CacheConfig,
                 key_cache_config: List[ov.PartialShape],
                 value_cache_config: List[ov.PartialShape],
                 model_config: ModelConfig, parallel_config: ParallelConfig,
                 device_config: DeviceConfig, ov_core: ov.Core,
                 ov_device: str) -> None:
        assert device_config.device_type == "cpu"
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.key_cache_config = key_cache_config
        self.value_cache_config = value_cache_config
        self.num_layers = len(value_cache_config)
        self.block_size = cache_config.block_size
        # vLLM names device-resident blocks num_gpu_blocks even when the
        # selected OpenVINO target is a CPU.
        self.num_device_blocks = cache_config.num_gpu_blocks
        self.num_swap_blocks = cache_config.num_cpu_blocks
        self.attn_backend = OpenVINOAttentionBackend()
        self.ov_cache_dtype = str_to_ov_type[cache_config.cache_dtype]
        self.kv_cache = self._allocate_kv_cache(self.num_device_blocks,
                                                ov_core, ov_device)
        self.swap_cache = self._allocate_swap_cache(self.num_swap_blocks,
                                                     ov_device)

    def _allocate_kv_cache(self, num_blocks: int, ov_core: ov.Core,
                           ov_device: str) -> List[Tuple[ov.Tensor, ov.Tensor]]:
        kv_cache = []
        for key_shape, value_shape in zip(self.key_cache_config,
                                          self.value_cache_config):
            key_shape[0] = num_blocks
            value_shape[0] = num_blocks
            if current_platform.is_openvino_cpu():
                key_blocks = ov.Tensor(self.ov_cache_dtype,
                                       key_shape.to_shape())
                value_blocks = ov.Tensor(self.ov_cache_dtype,
                                         value_shape.to_shape())
            else:
                context = ov_core.get_default_context(ov_device)
                key_blocks = context.create_tensor(
                    self.ov_cache_dtype, key_shape.to_shape(), {})
                value_blocks = context.create_tensor(
                    self.ov_cache_dtype, value_shape.to_shape(), {})
            kv_cache.append((key_blocks, value_blocks))
        return kv_cache

    def _allocate_swap_cache(self, num_blocks: int, ov_device: str):
        if num_blocks == 0:
            return []
        assert not current_platform.is_openvino_cpu()
        swap_cache = []
        for key_shape, value_shape in zip(self.key_cache_config,
                                          self.value_cache_config):
            key_shape[0] = num_blocks
            value_shape[0] = num_blocks
            swap_cache.append((
                ov.Tensor(self.ov_cache_dtype, key_shape.to_shape()),
                ov.Tensor(self.ov_cache_dtype, value_shape.to_shape()),
            ))
        return swap_cache

    def swap_in(self, src_to_dst):
        for swap_tensors, kv_tensors in zip(self.swap_cache, self.kv_cache):
            for swap_tensor, kv_tensor in zip(swap_tensors, kv_tensors):
                self.attn_backend.swap_blocks(swap_tensor, kv_tensor,
                                              src_to_dst)

    def swap_out(self, src_to_dst):
        for swap_tensors, kv_tensors in zip(self.swap_cache, self.kv_cache):
            for swap_tensor, kv_tensor in zip(swap_tensors, kv_tensors):
                self.attn_backend.swap_blocks(kv_tensor, swap_tensor,
                                              src_to_dst)

    def copy(self, src_to_dsts):
        if src_to_dsts:
            self.attn_backend.copy_blocks(self.kv_cache, src_to_dsts)

    @staticmethod
    def get_cache_block_size(cache_dtype, key_cache_config,
                             value_cache_config) -> int:
        total_elements = 0
        for key_shape, value_shape in zip(key_cache_config,
                                          value_cache_config):
            total_elements += (key_shape[1].get_length()
                               * key_shape[2].get_length()
                               * key_shape[3].get_length())
            total_elements += (value_shape[1].get_length()
                               * value_shape[2].get_length()
                               * value_shape[3].get_length())
        return str_to_ov_type[cache_dtype].size * total_elements

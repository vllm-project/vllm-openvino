# SPDX-License-Identifier: Apache-2.0

from typing import Tuple
from vllm.logger import init_logger
logger = init_logger(__name__)


def determine_num_available_blocks(current_platform,  cache_config, cache_block_size, profile_run_func) -> Tuple[int, int]:
    """Determine the number of blocks available for the KV cache.

    This determines how many KV blocks can fit into the configured
    KV cache space.
    """
    kvcache_space_bytes = cache_config.openvino_kvcache_space_bytes

    if current_platform.is_openvino_cpu():
        num_device_blocks = int(kvcache_space_bytes // cache_block_size)
        num_swap_blocks = 0
    else:
        if kvcache_space_bytes > 0:
            logger.info("KV_CACHE size was explicitly configured via "
                        "VLLM_OPENVINO_KVCACHE_SPACE environment "
                        "variable, ignoring profiling run.")
            kv_cache_size = kvcache_space_bytes
        else:
            try:
                kv_cache_size = profile_run_func()
            except Exception as err:
                raise RuntimeError(
                    "The error occurred during profile run. This might be "
                    "due to insufficient GPU memory. Consider decreasing "
                    "`max_model_len` to limit the maximum simultaneously "
                    "processed tokens.") from err

        num_device_blocks = int(kv_cache_size // cache_block_size)
        num_swap_blocks = int(cache_config.swap_space_bytes //
                              cache_block_size)

    return num_device_blocks, num_swap_blocks

def get_max_allocatable_memory_gpu(ov_core, ov_device, key_cache_config, value_cache_config):
    import openvino.properties.intel_gpu as intel_gpu
    if not hasattr(intel_gpu, "device_max_alloc_mem_size"):
        import sys
        return sys.maxsize

    max_tensor_alloc_size_gpu = ov_core.get_property(ov_device, intel_gpu.device_max_alloc_mem_size)
    assert len(key_cache_config) == len(value_cache_config), "Key cache config length should be equal to value cache config length."
    return len(key_cache_config) * 2 * max_tensor_alloc_size_gpu

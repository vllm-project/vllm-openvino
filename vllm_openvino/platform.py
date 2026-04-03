# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Optional

import torch

from vllm.logger import init_logger
from vllm.platforms.interface import Platform, PlatformEnum

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

logger = init_logger(__name__)

try:
    import openvino as ov
except ImportError as e:
    logger.warning("Failed to import OpenVINO with %r", e)
    
class OpenVinoPlatform(Platform):
    # OOT (Out-Of-Tree) refers to hardware platforms that are integrated with vLLM
    # through plugin projects, rather than being part of the vLLM core.
    _enum = PlatformEnum.OOT
    device_name: str = "openvino"
    device_type: str = "cpu" 
    
    @classmethod
    def get_attn_backend_cls(cls,
                            selected_backend: "AttentionBackendEnum",
                            attn_selector_config: "AttentionSelectorConfig") -> str:
        logger.info("Using OpenVINO Attention backend.")
        return "vllm_openvino.attention.backends.openvino.OpenVINOAttentionBackend"

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "openvino"

    @classmethod
    def inference_mode(cls):
        return torch.inference_mode(mode=True)

    @classmethod
    def is_openvino_cpu(cls) -> bool:
        return True

    @classmethod
    def is_openvino_gpu(cls) -> bool:
        return False

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        logger.warning("Pin memory is not supported on OpenVINO.")
        return False

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        GiB_bytes = 1_073_741_824

        parallel_config = vllm_config.parallel_config
        openvino_config = vllm_config.additional_config.get("openvino", {})
        
        assert (parallel_config.world_size == 1
                ), "OpenVINO only supports single CPU socket currently."

        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = \
                "vllm_openvino.worker_v1.openvino_worker_v1.OpenVINOWorkerV1"
        # check and update model config
        model_config = vllm_config.model_config
        if not model_config.enforce_eager:
            logger.warning(
                "CUDA graph is not supported on OpenVINO backend, fallback to "
                "the eager mode.")
            model_config.enforce_eager = True

        cache_config = vllm_config.cache_config
        if cache_config and cache_config.block_size is None:
            # for openVINO cpu will be updated to 32
            cache_config.block_size = 16

       
        cache_config.cache_dtype = "dynamic"
        
        logger.info("OpenVINO runtime config: %s", openvino_config)

        assert OpenVinoPlatform.is_openvino_cpu()
        if cache_config.block_size != 32:
            logger.info(
                f"OpenVINO CPU optimal block size is 32, overriding currently set {cache_config.block_size}"  # noqa: G004, E501
            )
            cache_config.block_size = 32

        kv_cache_space = openvino_config.get("VLLM_OPENVINO_KVCACHE_SPACE", 0)
        if kv_cache_space >= 0:
            if kv_cache_space == 0 and OpenVinoPlatform.is_openvino_cpu():
                cache_config.openvino_kvcache_space_bytes = 4 * GiB_bytes  # type: ignore
                logger.warning(
                    "Environment variable VLLM_OPENVINO_KVCACHE_SPACE (GB) "
                    "for OpenVINO backend is not set, using 4 by default.")
            else:
                cache_config.openvino_kvcache_space_bytes = (  # type: ignore
                    kv_cache_space * GiB_bytes)
        else:
            raise RuntimeError(
                "Invalid environment variable VLLM_OPENVINO_KVCACHE_SPACE"
                f" {kv_cache_space}, expect a positive integer value.")

        #assert vllm_config.device_config.device_type == "openvino" # see above, device_type!
        assert vllm_config.device_config.device_type == "cpu"
        assert vllm_config.lora_config is None, \
            "OpenVINO backend doesn't support LoRA"
        assert cls.is_openvino_cpu() or \
            cls.is_openvino_gpu(), \
            "OpenVINO backend supports only CPU and GPU devices"
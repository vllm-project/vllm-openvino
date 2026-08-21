# SPDX-License-Identifier: Apache-2.0

import contextlib
from typing import TYPE_CHECKING, Optional

import torch
from vllm.logger import init_logger
from vllm.platforms.interface import Platform, PlatformEnum

import vllm_openvino.envs as envs  # not sure if this is a optimal solution!

if TYPE_CHECKING:
    from vllm.config import VllmConfig, ModelConfig
else:
    VllmConfig = None
    ModelConfig = None

logger = init_logger(__name__)

try:
    import openvino as ov
    import openvino.properties.hint as hints
except ImportError as e:
    logger.warning("Failed to import OpenVINO with %r", e)


class OpenVinoPlatform(Platform):
    # vLLM-side tensors and sampling stay on CPU. VLLM_OPENVINO_DEVICE selects
    # the actual OpenVINO CPU or GPU execution target.
    _enum = PlatformEnum.CPU
    device_name: str = "openvino"
    device_type: str = "cpu"

    @classmethod
    def get_attn_backend_cls(cls, selected_backend, attn_selector_config,
                             num_heads: int | None = None) -> str:
        logger.info("Using OpenVINO Attention backend.")
        return "vllm_openvino.attention.backends.openvino.OpenVINOAttentionBackend"

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "openvino"

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        return False

    @classmethod
    def inference_mode(cls):
        return torch.inference_mode(mode=True)

    @classmethod
    def is_openvino_cpu(cls) -> bool:
        return "CPU" in envs.VLLM_OPENVINO_DEVICE

    @classmethod
    def is_openvino_gpu(cls) -> bool:
        return "GPU" in envs.VLLM_OPENVINO_DEVICE

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        logger.warning("Pin memory is not supported on OpenVINO.")
        return False

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        torch.manual_seed(seed)

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        from vllm.utils.mem_constants import GiB_bytes

        parallel_config = vllm_config.parallel_config
        assert (parallel_config.world_size == 1
                ), "OpenVINO only supports single CPU socket currently."

        if vllm_config.scheduler_config.async_scheduling:
            logger.warning("Async scheduling is not supported on OpenVINO.")
            vllm_config.scheduler_config.async_scheduling = False

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

        # check and update cache config
        ov_core = ov.Core()
        cache_config = vllm_config.cache_config
        if cache_config and cache_config.block_size is None:
            cache_config.block_size = 16

        if envs.VLLM_OPENVINO_KV_CACHE_PRECISION == "u8":
            logger.info("KV cache type is overridden to u8 via "
                        "VLLM_OPENVINO_KV_CACHE_PRECISION env var.")
            cache_config.cache_dtype = "u8"
        elif envs.VLLM_OPENVINO_KV_CACHE_PRECISION == "i8":
            logger.info("KV cache type is overridden to i8 via "
                        "VLLM_OPENVINO_KV_CACHE_PRECISION env var.")
            cache_config.cache_dtype = "i8"
        elif envs.VLLM_OPENVINO_KV_CACHE_PRECISION == "f16" or envs.VLLM_OPENVINO_KV_CACHE_PRECISION == "fp16":
            logger.info("KV cache type is overridden to fp16 via "
                        "VLLM_OPENVINO_KV_CACHE_PRECISION env var.")
            cache_config.cache_dtype = "f16"
        elif envs.VLLM_OPENVINO_KV_CACHE_PRECISION == "bf16":
            logger.info("KV cache type is overridden to bp16 via "
                        "VLLM_OPENVINO_KV_CACHE_PRECISION env var.")
            cache_config.cache_dtype = "bf16"
        elif envs.VLLM_OPENVINO_KV_CACHE_PRECISION == "fp32" or envs.VLLM_OPENVINO_KV_CACHE_PRECISION == "f32":
            logger.info("KV cache type is overridden to f16 via "
                        "VLLM_OPENVINO_KV_CACHE_PRECISION env var.")
            cache_config.cache_dtype = "f32"
        else:
            logger.info("KV cache type is not specified via "
                        "VLLM_OPENVINO_KV_CACHE_PRECISION env var. "
                        "It will be determined automatically by a plugin")
            cache_config.cache_dtype = "dynamic"

        if OpenVinoPlatform.is_openvino_cpu():
            if cache_config.block_size != 32:
                logger.info(
                    f"OpenVINO CPU optimal block size is 32, overriding currently set {cache_config.block_size}"  # noqa: G004, E501
                )
                cache_config.block_size = 32
        else:
            if cache_config.block_size != 16:
                logger.info(
                    f"OpenVINO GPU optimal block size is 16, overriding currently set {cache_config.block_size}"  # noqa: G004, E501
                )
                cache_config.block_size = 16

        kv_cache_space = envs.VLLM_OPENVINO_KVCACHE_SPACE
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

        assert vllm_config.device_config.device_type == "cpu"
        assert vllm_config.lora_config is None, \
            "OpenVINO backend doesn't support LoRA"
        assert cls.is_openvino_cpu() or \
            cls.is_openvino_gpu(), \
            "OpenVINO backend supports only CPU and GPU devices"

    @classmethod
    def import_kernels(cls) -> None:
        ignored_msg = "dynamic module does not define module export function"
        if torch.cpu._is_avx512_supported():
            module_name = ("vllm._C" if torch.cpu._is_avx512_bf16_supported()
                           else "vllm._C_AVX512")
        else:
            module_name = "vllm._C_AVX2"
        try:
            __import__(module_name)
        except ModuleNotFoundError:
            # Native vLLM kernels are intentionally absent for the empty target.
            pass
        except ImportError as error:
            if ignored_msg not in str(error):
                logger.warning_once("Failed to import %s: %r", module_name,
                                    error)
        with contextlib.suppress(ImportError):
            import vllm._moe_C_stable_libtorch  # noqa: F401

    @classmethod
    def supports_v1(cls, model_config: ModelConfig) -> bool:
        """Returns whether the current platform can support v1 for the supplied
        model configuration.
        """
        return True

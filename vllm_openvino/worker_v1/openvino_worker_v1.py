# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional, Tuple, Set

import openvino as ov
import torch
import torch.nn as nn
from vllm.config import VllmConfig

from vllm.logger import init_logger
from vllm.platforms import current_platform
# from vllm.utils import bind_kv_cache
from vllm.v1.kv_cache_interface import KVCacheSpec, KVCacheConfig, FullAttentionSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.worker_base import WorkerBase
from vllm.utils.torch_utils import set_random_seed
from vllm.tasks import GenerationTask, SupportedTask

from vllm_openvino.worker_v1.openvino_model_runner_v1 import OpenVINOModelRunnerV1
from vllm_openvino.utils import determine_num_available_blocks

logger = init_logger(__name__)

str_to_torch_type = {
    "u8": torch.uint8,
    "i8": torch.int8,
    "fp16": torch.float16,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
    "fp32": torch.float32
}


class OpenVINOWorkerV1(WorkerBase):
    """A worker class that executes the model on OpenVINO backend.

    Each worker is associated with a single OpenVINO device. The worker is
    responsible for maintaining the KV cache and executing the model on the
    OpenVINO backend.
    """

    def __init__(
            self,
            vllm_config: VllmConfig,
            local_rank: int,
            rank: int,
            distributed_init_method: str,
            is_driver_worker: bool = False,
    ):
        super().__init__(vllm_config=vllm_config,
                         local_rank=local_rank,
                         rank=rank,
                         distributed_init_method=distributed_init_method,
                         is_driver_worker=is_driver_worker)
        self.ov_core = ov.Core()
                
        if self.is_driver_worker:
            assert self.rank == 0, "The driver worker must have rank 0."

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()
            
        self.model_runner = OpenVINOModelRunnerV1(
            self.ov_core,
            vllm_config=self.vllm_config
        )
        # swap_blocks = 0 for OpenVINO CPU plugin
        self.num_swap_blocks = 0
        
    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        supported_tasks = list[GenerationTask]()
        supported_tasks.append("generate")
        return supported_tasks

    def init_device(self) -> None:
        # Set random seed.
        set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()

        # we need to take information about KV cache config from compiled model
        compiled_model = self.model_runner.get_model().ov_request.get_compiled_model()

        self.key_cache_config = []
        self.value_cache_config = []

        for input_port in compiled_model.inputs:
            input_name = input_port.get_any_name()

            if input_name.startswith("key_cache."):
                self.cache_dtype = input_port.get_element_type().to_string()
                self.key_cache_config.append(input_port.get_partial_shape())
            if input_name.startswith("value_cache."):
                self.value_cache_config.append(input_port.get_partial_shape())

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Initialize the KV cache.

        For CPU, we use the num_gpu_blocks to
        determine how many non-swappable CPU blocks to allocate.
        """

        num_device_blocks = num_gpu_blocks
        num_swap_blocks = num_cpu_blocks

        if current_platform.is_openvino_cpu():
            assert (num_swap_blocks == 0
                    ), f"{type(self)} does not support swappable cache for CPU"

        self._validate_num_blocks(num_device_blocks)
        self.cache_config.num_gpu_blocks = num_device_blocks
        self.cache_config.num_cpu_blocks = num_swap_blocks

        # Initialize the cache.
        self._init_cache_engine()

    def _validate_num_blocks(self, num_blocks: int) -> None:
        """Raise errors if the num_blocks is invalid."""
        if num_blocks <= 0:
            raise ValueError(
                "No available memory for the cache blocks. "
                "Try increasing `VLLM_OPENVINO_KVCACHE_SPACE` when "
                "initializing the engine.")

        max_seq_len = self.cache_config.block_size * num_blocks
        if self.model_config.max_model_len > max_seq_len:
            raise ValueError(
                f"The model's max seq len ({self.model_config.max_model_len}) "
                "is larger than the maximum number of tokens that can be "
                f"stored in KV cache ({max_seq_len}). Try increasing "
                "`VLLM_OPENVINO_KVCACHE_SPACE` or decreasing `max_model_len` "
                "when initializing the engine.")

    def _init_cache_engine(self) -> None:
        # we need to override precision in self.cache_config to one, inference during compile_model
        self.cache_config.cache_dtype = self.cache_dtype

        self.cache_engine = OpenVINOCacheEngine(
            self.vllm_config,
            self.key_cache_config,
            self.value_cache_config,
        )
        self.kv_cache = self.cache_engine.kv_cache
        self.model_runner.block_size = self.cache_engine.block_size

        assert self.kv_cache is not None

    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()

    def execute_model(
        self,
        execute_model_req = None,
    ) -> ModelRunnerOutput:
        if execute_model_req.total_num_scheduled_tokens == 0:
            return ModelRunnerOutput(
                req_ids=[],
                req_id_to_index={},
                sampled_token_ids=[],
                logprobs=None,
                prompt_logprobs_dict={},
            )
        return self.model_runner.execute_model(execute_model_req, self.kv_cache)
    
    def sample_tokens(
        self, grammar_output: "GrammarOutput"
    ) :
        """Should be called immediately after execute_model iff it returned None."""
        ''' TODO: Need to move the sample logic from execute model here in sample_tokens'''
        return self.model_runner._model_ouput

    def get_cache_block_size_bytes(self) -> int:
        """Return the size in bytes of a single KV cache block."""
        return OpenVINOCacheEngine.get_cache_block_size(
            self.cache_config.cache_dtype,
            self.key_cache_config,
            self.value_cache_config,
        )

    def profile_run(self) -> int:
        raise NotImplementedError("CPU device isn't supposed to use profile run.")
        

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """Get specifications for KV cache implementation."""
        key_cache_config = self.key_cache_config
        value_cache_config = self.value_cache_config
        block_size = self.cache_config.block_size
        cache_type = self.cache_dtype
        assert cache_type in str_to_torch_type.keys(), "Unexpected cache type {}".format(cache_type)
        kv_cache_spec = {}

        for idx, (key_cache_shape, value_cache_shape) in enumerate(zip(key_cache_config, value_cache_config)):
            # This shape is used for calculation of max memory required by KV-cache
            kv_cache_spec["{}".format(idx)] = FullAttentionSpec(block_size=block_size,
                                                                num_kv_heads=max(key_cache_shape[1].get_length(),
                                                                                 value_cache_shape[1].get_length()),
                                                                head_size=max(key_cache_shape[3].get_length(),
                                                                              value_cache_shape[3].get_length()),
                                                                dtype=str_to_torch_type[cache_type],)
        return kv_cache_spec

    def determine_available_memory(self) -> int:
        """Determines how much memory is needed for KV-cache
        """
        self.cache_config.cache_dtype = self.cache_dtype
        # For OpenVINO backend, in case of CPU device, the block number will be
        # calculated based on the openvino_kvcache_space_bytes.
        cache_block_size = self.get_cache_block_size_bytes()
        num_device_blocks, num_swap_blocks = determine_num_available_blocks(current_platform,
                                                                            self.cache_config,
                                                                            cache_block_size,
                                                                            self.profile_run)
        self.num_swap_blocks = num_swap_blocks
        return num_device_blocks * cache_block_size

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate NPU KV cache with the specified kv_cache_config."""
        self.initialize_cache(kv_cache_config.num_blocks, self.num_swap_blocks)

    def compile_or_warm_up_model(self) -> None:
        # Compile is performed on model loading stage
        pass

    def list_loras(self) -> Set[int]:
        raise NotImplementedError("LoRA is not supported.")

    def pin_lora(self, lora_id: int) -> bool:
        raise NotImplementedError("LoRA is not supported.")

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError("LoRA is not supported.")

    def add_lora(self, lora_request) -> bool:
        raise NotImplementedError("LoRA is not supported.")

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        return self.kv_cache_config.num_blocks


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
    """Manages the KV cache for OpenVINO backend.

    This class is responsible for initializing and managing CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as copying.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        key_cache_config: dict,
        value_cache_config: List[ov.PartialShape],  # to rename dict
    ) -> None:
        assert vllm_config.device_config.device_type == "cpu"
        cache_config = vllm_config.cache_config

        self.key_cache_config = key_cache_config
        self.value_cache_config = value_cache_config
        self.num_layers = len(self.value_cache_config)

        self.block_size = cache_config.block_size
        # Note: In CacheConfig, num_gpu_blocks actual is num_cpu_blocks
        # for OpenVINO backend with a CPU target device, because we want
        # to reuse KV cache management in the scheduler.
        self.num_device_blocks = cache_config.num_gpu_blocks
        self.num_swap_blocks = cache_config.num_cpu_blocks


        self.ov_cache_dtype = str_to_ov_type[cache_config.cache_dtype]

        # Initialize the cache.
        self.kv_cache: List[Tuple[ov.Tensor,
                                  ov.Tensor]] = self._allocate_kv_cache(
                                      self.num_device_blocks)


    def _allocate_kv_cache(
        self,
        num_blocks: int,
    ) -> List[Tuple[ov.Tensor, ov.Tensor]]:
        """Allocates KV cache."""
        kv_cache: List[Tuple[ov.Tensor, ov.Tensor]] = []

        for key_cache_pshape, value_cache_pshape in zip(self.key_cache_config, self.value_cache_config):
            key_cache_shape = key_cache_pshape
            value_cache_shape = value_cache_pshape
            key_cache_shape[0] = num_blocks
            value_cache_shape[0] = num_blocks
            key_cache_shape = key_cache_shape.to_shape()
            value_cache_shape = value_cache_shape.to_shape()

            assert (current_platform.is_openvino_cpu())
            key_blocks = ov.Tensor(self.ov_cache_dtype, key_cache_shape)
            value_blocks = ov.Tensor(self.ov_cache_dtype, value_cache_shape)
            kv_cache.append((key_blocks, value_blocks))
        return kv_cache

    @staticmethod
    def get_cache_block_size(
        cache_dtype: str,
        key_cache_config: List[ov.PartialShape],
        value_cache_config: List[ov.PartialShape],
    ) -> int:
        total_elements = 0
        for key_cache_shape, value_cache_shape in zip(key_cache_config, value_cache_config):
             total_elements += key_cache_shape[1].get_length() * key_cache_shape[2].get_length() * key_cache_shape[3].get_length()
             total_elements += value_cache_shape[1].get_length() * value_cache_shape[2].get_length() * value_cache_shape[3].get_length()
        return str_to_ov_type[cache_dtype].size * total_elements
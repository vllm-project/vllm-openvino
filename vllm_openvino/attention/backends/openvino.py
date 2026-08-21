# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import openvino as ov
import torch

from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata


def create_roi_view(tensor: ov.Tensor, block_number: int) -> ov.Tensor:
    """Return a view of the given cache tensor limited to one KV block."""
    roi_begin = [0, 0, 0, 0]
    roi_end = list(tensor.get_shape())
    roi_begin[0] = block_number
    roi_end[0] = block_number + 1

    if isinstance(tensor, ov.Tensor):
        return ov.Tensor(tensor, roi_begin, roi_end)
    else:
        return ov.RemoteTensor(tensor, roi_begin, roi_end)


def copy_cache_block(src_tensor: ov.Tensor, dst_tensor: ov.Tensor,
                     src_offset: int, dst_offset: int) -> None:
    src_roi_tensor = create_roi_view(src_tensor, src_offset)
    dst_roi_tensor = create_roi_view(dst_tensor, dst_offset)
    src_roi_tensor.copy_to(dst_roi_tensor)


def zero_cache_block(tensor: ov.Tensor, block_number: int) -> None:
    """Fill one KV cache block with zeros."""
    roi_tensor = create_roi_view(tensor, block_number)
    if isinstance(tensor, ov.Tensor):
        roi_tensor.data.fill(0)
    else:
        # Remote tensors do not expose writable host memory, so zero a host
        # buffer and copy it into the target block instead.
        zero_tensor = ov.Tensor(tensor.get_element_type(),
                                list(roi_tensor.get_shape()))
        zero_tensor.data.fill(0)
        zero_tensor.copy_to(roi_tensor)


class OpenVINOAttentionBackend(AttentionBackend):

    # PagedAttention is embedded in the exported OpenVINO graph, so vLLM's
    # attention implementation and metadata builder are not used.

    @staticmethod
    def get_name() -> str:
        return "OPENVINO"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError

    @staticmethod
    def get_builder_cls():
        raise NotImplementedError

    @staticmethod
    def make_metadata(*args, **kwargs) -> "AttentionMetadata":
        raise NotImplementedError

    @staticmethod
    def make_openvino_metadata(*args, **kwargs) -> "OpenVINOAttentionMetadata":
        return OpenVINOAttentionMetadata(*args, **kwargs)

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> Tuple[int, ...]:
        return (2, num_blocks, num_kv_heads, block_size, head_size)

    @staticmethod
    def swap_blocks(
        src_tensor: ov.Tensor,
        dst_tensor: ov.Tensor,
        src_to_dists: List[Tuple[int, int]],
    ) -> None:
        for src, dst in src_to_dists:
            copy_cache_block(src_tensor, dst_tensor, src, dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[Tuple[ov.Tensor, ov.Tensor]],
        src_to_dists: List[Tuple[int, int]],
    ) -> None:
        for src, dst in src_to_dists:
            for key_cache, value_cache in kv_caches:
                copy_cache_block(key_cache, key_cache, src, dst)
                copy_cache_block(value_cache, value_cache, src, dst)

    @staticmethod
    def zero_blocks(
        kv_caches: List[Tuple[ov.Tensor, ov.Tensor]],
        block_ids: List[int],
    ) -> None:
        for block_id in block_ids:
            for key_cache, value_cache in kv_caches:
                zero_cache_block(key_cache, block_id)
                zero_cache_block(value_cache, block_id)


@dataclass
class OpenVINOAttentionMetadata:
    """Metadata for OpenVINOAttentionBackend.

    Basic terms used below:
    - batch_size_in_sequences - total number of sequences to execute​
    - prompt_lens – per sequence size number of scheduled tokens​
    - batch_size_in_tokens = sum(prompt_lens)​
    - max_context_len = max(context_lens)​
    - max_num_blocks = div_up(max_context_len / BLOCK_SIZE)​
    - num_blocks – total number of blocks in block_indices​
    """

    # Describes past KV cache size for each sequence within a batch
    # Shape: [batch_size_in_sequences]
    # Type: i32​
    past_lens: torch.Tensor

    # Describes start indices of input / speculative tokens from
    # current sequences within a batch sequence​
    # Shape: [batch_size_in_sequences + 1]​
    # Type: i32
    subsequence_begins: torch.Tensor

    # Describes block tables for each sequence within a batch​ -
    # indices along 0th dimension in key_cache and value_cache inputs​
    # Shape: [num_blocks]
    # Type: i32​
    block_indices: torch.Tensor

    # Describes block tables for each sequence within a batch​ -
    # for i-th element, it is an index in block_indices with the
    # first block belonging to i-th sequence​
    # Shape: [batch_size_in_sequences + 1]
    # Type: i32​
    block_indices_begins: torch.Tensor

    # Describes max context length
    # Shape: scalar
    # Type: i32
    max_context_len: torch.Tensor

    # The index maps that relate multi-modal embeddings to the corresponding
    # placeholders.
    #
    # N.B. These aren't really related to attention and don't belong on this
    # type -- this is just a temporary solution to make them available to
    # `model_executable`.
    multi_modal_placeholder_index_maps: Optional[Dict[str, int]]

    # Enable/disable KV scales calculation. This is so that we can disable the
    # calculation until after prefill and cuda graph capture.
    enable_kv_scales_calculation: bool

    # Indices of sampled tokens, used in V1 vLLM API only.
    sampled_token_indices: ov.Tensor

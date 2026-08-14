# SPDX-License-Identifier: Apache-2.0

from typing import List, NamedTuple, Optional

import torch

from vllm_openvino.attention.backends.openvino import OpenVINOAttentionMetadata


class ModelInput(NamedTuple):
    input_tokens: torch.Tensor
    input_positions: torch.Tensor
    attn_metadata: Optional[OpenVINOAttentionMetadata]
    seq_lens: List[int]
    query_lens: List[int]
    multi_modal_kwargs: Optional[dict] = None

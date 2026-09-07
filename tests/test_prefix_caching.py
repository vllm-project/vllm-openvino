# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import openvino as ov
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.core.sched.output import SchedulerOutput

from vllm_openvino.attention.backends.openvino import (
    OpenVINOAttentionBackend,
    copy_cache_block,
    zero_cache_block,
)
from vllm_openvino.worker_v1.openvino_worker_v1 import OpenVINOWorkerV1


def _make_cache_tensor(num_blocks: int = 4, fill: float = 7.0) -> ov.Tensor:
    tensor = ov.Tensor(ov.Type.f32, [num_blocks, 2, 8, 16])
    tensor.data.fill(fill)
    return tensor


def test_copy_cache_block_copies_block():
    src = _make_cache_tensor(fill=3.0)
    dst = _make_cache_tensor(fill=9.0)

    copy_cache_block(src, dst, 0, 2)

    assert (dst.data[2] == 3.0).all()
    assert (dst.data[0] == 9.0).all()
    assert (dst.data[1] == 9.0).all()


def test_zero_cache_block_zeroes_only_target_block():
    tensor = _make_cache_tensor(fill=7.0)

    zero_cache_block(tensor, 1)

    assert (tensor.data[1] == 0.0).all()
    assert (tensor.data[0] == 7.0).all()
    assert (tensor.data[2] == 7.0).all()


def test_zero_blocks_covers_all_layers():
    kv_caches = [
        (_make_cache_tensor(fill=1.0), _make_cache_tensor(fill=2.0)),
        (_make_cache_tensor(fill=3.0), _make_cache_tensor(fill=4.0)),
    ]

    OpenVINOAttentionBackend.zero_blocks(kv_caches, [0, 3])

    for key_cache, value_cache in kv_caches:
        assert (key_cache.data[0] == 0.0).all()
        assert (key_cache.data[3] == 0.0).all()
        assert (key_cache.data[1] == 1.0).any() or (key_cache.data[1] == 3.0).any()
        assert (value_cache.data[0] == 0.0).all()
        assert (value_cache.data[3] == 0.0).all()
        assert (value_cache.data[1] == 2.0).any() or (value_cache.data[1] == 4.0).any()


def test_execute_model_applies_prefix_caching_instructions(monkeypatch):
    calls = {"copy": [], "zero": []}
    cache_engine = SimpleNamespace(
        copy=lambda src_to_dsts: calls["copy"].extend(src_to_dsts),
        zero_blocks=lambda block_ids: calls["zero"].extend(block_ids))
    worker = object.__new__(OpenVINOWorkerV1)
    worker.__dict__["cache_engine"] = cache_engine

    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.kv_cache_block_copies = [
        KVCacheBlockCopy(src_block_id=1, dst_block_id=5),
        KVCacheBlockCopy(src_block_id=2, dst_block_id=6),
    ]
    scheduler_output.new_block_ids_to_zero = [7, 8]

    # Instructions must be applied even on steps with no scheduled tokens.
    output = worker.execute_model(scheduler_output)

    assert output.req_ids == []
    assert calls["copy"] == [(1, 5), (2, 6)]
    assert calls["zero"] == [7, 8]


def test_execute_model_skips_empty_instructions(monkeypatch):
    calls = []
    cache_engine = SimpleNamespace(
        copy=lambda src_to_dsts: calls.append("copy"),
        zero_blocks=lambda block_ids: calls.append("zero"))
    worker = object.__new__(OpenVINOWorkerV1)
    worker.__dict__["cache_engine"] = cache_engine

    scheduler_output = SchedulerOutput.make_empty()
    assert scheduler_output.kv_cache_block_copies is None
    assert scheduler_output.new_block_ids_to_zero is None

    worker.execute_model(scheduler_output)

    assert calls == []
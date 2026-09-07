# SPDX-License-Identifier: Apache-2.0

import inspect
from types import SimpleNamespace

from vllm.v1.outputs import ModelRunnerOutput

from vllm_openvino.attention.backends.openvino import OpenVINOAttentionBackend
from vllm_openvino.worker_v1 import openvino_worker_v1
from vllm_openvino.worker_v1.openvino_worker_v1 import OpenVINOWorkerV1


def test_attention_backend_is_concrete():
    assert not inspect.isabstract(OpenVINOAttentionBackend)


def test_empty_model_runner_output_uses_current_api():
    output = ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        sampled_token_ids=[],
        logprobs=None,
        prompt_logprobs_dict={},
    )
    assert output.sampled_token_ids == []


def test_worker_reports_generation_support():
    worker = object.__new__(OpenVINOWorkerV1)
    assert worker.get_supported_tasks() == ("generate",)


def test_worker_warms_sampler_and_resets_seed(monkeypatch):
    calls = []
    worker = object.__new__(OpenVINOWorkerV1)
    worker.__dict__["model_runner"] = SimpleNamespace(
        warm_up_sampler=lambda: calls.append("warmup"))
    worker.__dict__["model_config"] = SimpleNamespace(seed=42)
    monkeypatch.setattr(openvino_worker_v1, "set_random_seed",
                        lambda seed: calls.append(("seed", seed)))

    worker.compile_or_warm_up_model()

    assert calls == ["warmup", ("seed", 42)]

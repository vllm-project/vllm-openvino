# SPDX-License-Identifier: Apache-2.0

"""Integration test for prefix caching on the OpenVINO backend.

Runs an engine with prefix caching enabled and disabled, generates greedy
outputs for prompts sharing a long common prefix, and verifies:

  1. outputs are identical with and without prefix caching (correctness --
     the cached KV data must be bit-exact for greedy decoding to match),
  2. the V1 scheduler actually reuses the cached prefix (the enabled run
     schedules far fewer tokens overall than the disabled run).

The engine runs in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0) so the worker
can be instrumented from this module to observe scheduler instructions.

Run directly:

    python tests/test_prefix_caching_integration.py

Run via pytest (unit tests only, no model download):

    python -m pytest tests/test_prefix_caching.py
"""

import argparse
import os
import sys

# Run the V1 engine in-process so the worker instrumentation below observes
# scheduler instructions from the same process.
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, SamplingParams

from vllm_openvino.worker_v1.openvino_worker_v1 import OpenVINOWorkerV1

LONG_SENTENCE = (
    "The OpenVINO toolkit is an open-source toolkit for optimizing and "
    "inferencing AI workloads on Intel hardware, and it supports both CPU "
    "and GPU execution, along with a rich set of quantization tools. "
)

# Observed counters, populated by the instrumented worker while an engine runs.
stats = {"copies": 0, "zeros": 0, "scheduled": [], "call": None}
_orig_execute_model = OpenVINOWorkerV1.execute_model
_orig_generate = LLM.generate


def _execute_model(self, scheduler_output):
    copies = getattr(scheduler_output, "kv_cache_block_copies", None) or []
    zeros = getattr(scheduler_output, "new_block_ids_to_zero", None) or []
    stats["copies"] += len(copies)
    stats["zeros"] += len(zeros)
    total = getattr(scheduler_output, "total_num_scheduled_tokens", 0)
    if stats["call"] is not None:
        stats["scheduled"][stats["call"]] += total
    return _orig_execute_model(self, scheduler_output)


def _generate(self, prompts, sampling_params=None, use_tqdm=True, **kwargs):
    stats["scheduled"].append(0)
    stats["call"] = len(stats["scheduled"]) - 1
    try:
        return _orig_generate(self, prompts, sampling_params=sampling_params,
                              use_tqdm=False, **kwargs)
    finally:
        stats["call"] = None


OpenVINOWorkerV1.execute_model = _execute_model
LLM.generate = _generate


def build_prompts(model: str, prefix_tokens: int) -> tuple[str, str]:
    """Build two prompts sharing an exact token prefix."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model)
    prefix_ids = tokenizer(LONG_SENTENCE * 64,
                           add_special_tokens=True)["input_ids"][:prefix_tokens]
    prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=True)
    return (prefix_text + " What is the capital of France?",
            prefix_text + " What is the capital of Japan?")


def run(model: str, prompts: tuple[str, str], prefix_caching: bool,
        max_tokens: int) -> dict:
    stats["copies"] = 0
    stats["zeros"] = 0
    stats["scheduled"] = []
    stats["call"] = None
    llm = LLM(
        model=model,
        enable_prefix_caching=prefix_caching,
        enforce_eager=True,
        max_model_len=2048,
    )
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    # Sequential: the second prompt reuses the first prompt's prefix.
    first = llm.generate([prompts[0]], sampling_params, use_tqdm=False)[0]
    second = llm.generate([prompts[1]], sampling_params, use_tqdm=False)[0]
    # Concurrent: both prompts share the prefix in one batch.
    batch = llm.generate(list(prompts), sampling_params, use_tqdm=False)
    del llm
    return {
        "a": first.outputs[0].token_ids,
        "b": second.outputs[0].token_ids,
        "batch": [out.outputs[0].token_ids for out in batch],
        "copies": stats["copies"],
        "zeros": stats["zeros"],
        "scheduled": list(stats["scheduled"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--prefix-tokens", type=int, default=403)
    parser.add_argument("--max-tokens", type=int, default=24)
    args = parser.parse_args()

    prompts = build_prompts(args.model, args.prefix_tokens)

    print(f"PREFIX_TOKENS={args.prefix_tokens}", flush=True)
    on = run(args.model, prompts, prefix_caching=True,
             max_tokens=args.max_tokens)
    off = run(args.model, prompts, prefix_caching=False,
              max_tokens=args.max_tokens)

    print("ON :", on["copies"], on["zeros"], on["scheduled"], flush=True)
    print("OFF:", off["copies"], off["zeros"], off["scheduled"], flush=True)

    failures = []

    # Prefix caching must not change decoding results: the cached KV blocks
    # are copied byte-for-byte, so greedy argmax is identical.
    if (on["a"] != off["a"] or on["b"] != off["b"]
            or on["batch"] != off["batch"]):
        failures.append("prefix caching changed generation outputs")

    # Reusing the cached prefix means the enabled run schedules far fewer
    # tokens overall (the long shared prefix is not recomputed).
    reuse = off["scheduled"][0] + off["scheduled"][1] + off["scheduled"][2] \
        - on["scheduled"][0] - on["scheduled"][1] - on["scheduled"][2]
    print(f"PREFIX_REUSE_TOKENS={reuse}", flush=True)
    if reuse < args.prefix_tokens:
        failures.append(
            f"expected prefix reuse of >= {args.prefix_tokens} tokens, got "
            f"{reuse}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", flush=True)
        print("RESULT_FAIL", flush=True)
        return 1

    print("RESULT_OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

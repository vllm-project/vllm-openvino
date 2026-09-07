#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib.metadata
import json
import statistics
import time

from vllm import LLM, SamplingParams


PROMPTS = [
    "The future of AI is",
    "OpenVINO accelerates",
    "Machine learning models can",
    "The quick brown fox",
]


def get_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark repeatable offline OpenVINO generation.")
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repetitions < 1:
        raise ValueError("--repetitions must be at least 1")

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        ignore_eos=True,
    )

    start = time.perf_counter()
    llm = LLM(model=args.model)
    engine_init_seconds = time.perf_counter() - start

    runs = []
    for iteration in range(1, args.repetitions + 1):
        start = time.perf_counter()
        outputs = llm.generate(PROMPTS, sampling_params, use_tqdm=False)
        elapsed = time.perf_counter() - start
        output_tokens = sum(
            len(output.outputs[0].token_ids) for output in outputs)
        throughput = output_tokens / elapsed
        runs.append(throughput)
        print(f"RUN {iteration}: {throughput:.2f} output tokens/s")

    steady_state_runs = runs[1:] if len(runs) > 1 else runs
    result = {
        "model": args.model,
        "versions": {
            "vllm": get_version("vllm"),
            "vllm-openvino": get_version("vllm-openvino"),
            "openvino": get_version("openvino"),
            "optimum-intel": get_version("optimum-intel"),
        },
        "num_prompts": len(PROMPTS),
        "max_tokens": args.max_tokens,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        "engine_init_seconds": round(engine_init_seconds, 3),
        "throughput_tokens_per_second": [round(run, 2) for run in runs],
        "first_request_tokens_per_second": round(runs[0], 2),
        "steady_state_median_tokens_per_second": round(
            statistics.median(steady_state_runs), 2),
    }
    print("RESULT " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

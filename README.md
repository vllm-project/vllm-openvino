# vLLM OpenVINO Plugin

The vLLM OpenVINO plugin provides text generation for supported decoder-only
causal language models on x86-64 CPUs with AVX2 support and on integrated and
discrete Intel GPUs.

> [!NOTE]
> There are no pre-built plugin wheels or images. Install or build the plugin
> from source.

## Installation

## Requirements

- OS: Linux
- Python 3.10 through 3.14.
- x86-64 CPU with at least AVX2 support.
- vLLM 0.26.0.
- OpenVINO 2026.3 or newer.
- Optimum Intel 2.1.x.

## Set up using Python

### Pre-built wheels

Currently, there are no pre-built `vllm-openvino` wheels.

### Build wheel from source

First, install Python and ensure you have the latest pip. For example, on Ubuntu 22.04, you can run:

```console
sudo apt-get update  -y
sudo apt-get install python3-pip
pip install --upgrade pip
```

Second, clone vllm-openvino and install prerequisites for the vLLM OpenVINO backend installation:

```console
git clone https://github.com/vllm-project/vllm-openvino.git
cd vllm-openvino
```

Finally, install vLLM with OpenVINO backend:

```console
VLLM_TARGET_DEVICE="empty" PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu" python -m pip install -v .
```

> [!NOTE]
> To use an Intel GPU, follow the current OpenVINO
> [GPU device setup documentation](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html).

## Set up using Docker

### Pre-built images

Currently, there are no pre-built OpenVINO images.

### Build image from source

```console
docker build . -t vllm-openvino-env .
docker run -it --rm vllm-openvino-env
```

## Usage

Run the offline example:

```console
VLLM_OPENVINO_KVCACHE_SPACE=4 \
python3 examples/offline_inference_openvino.py
```

Start an OpenAI-compatible server:

```console
VLLM_OPENVINO_KVCACHE_SPACE=4 \
vllm serve facebook/opt-125m
```

## Supported features

The OpenVINO backend uses the vLLM V1 engine and supports:

- Offline and OpenAI-compatible text generation.
- Chunked prefill (`--enable-chunked-prefill`)

> [!NOTE]
> Prefix caching is temporarily disabled while its cache-copy path is updated
> for the current vLLM V1 scheduler.

## Performance tips

### vLLM OpenVINO backend environment variables

- `VLLM_OPENVINO_DEVICE` selects the inference device. The default is `CPU`.
  Use an indexed device such as `GPU.1` when multiple GPUs are available.
- `VLLM_OPENVINO_ENABLE_QUANTIZED_WEIGHTS=ON` enables 8-bit weight
  compression while exporting a Hugging Face model. Compression is disabled
  by default. To use another compression scheme, export the model with
  `optimum-cli` and pass the exported directory as the model ID.

### CPU performance tips

CPU uses the following environment variables to control behavior:

- `VLLM_OPENVINO_KVCACHE_SPACE` sets the KV cache size in GiB. CPU defaults
  to 4 GiB. A larger cache permits more concurrent or longer requests.
- `VLLM_OPENVINO_KV_CACHE_PRECISION` overrides the automatically selected KV
  cache precision. Supported values include `u8`, `i8`, `f16`, `bf16`, and
  `f32`.

To balance time to first token and inter-token latency, tune chunked prefill
with `--max-num-batched-tokens`. The best value depends on the model and CPU;
`256` is a useful starting point.

Example CPU throughput command:

```console
VLLM_OPENVINO_KVCACHE_SPACE=100 \
VLLM_OPENVINO_KV_CACHE_PRECISION=u8 \
VLLM_OPENVINO_ENABLE_QUANTIZED_WEIGHTS=ON \
vllm bench throughput \
    --model meta-llama/Llama-2-7b-chat-hf \
    --dataset-name random \
    --num-prompts 256 \
    --random-input-len 128 \
    --random-output-len 128 \
    --max-num-batched-tokens 256
```

Run `vllm bench throughput --help` to select a dataset and workload for a
production-scale benchmark.

### Upgrade benchmark

`benchmarks/benchmark_offline.py` is a small, version-neutral regression
benchmark. Run the same command from environments containing the old and new
plugin revisions:

```console
VLLM_OPENVINO_KVCACHE_SPACE=1 \
python3 benchmarks/benchmark_offline.py --repetitions 3
```

The benchmark reports engine initialization separately from first-request and
steady-state output throughput. It uses four prompts, 16 output tokens per
prompt, and random sampling by default. Treat results as comparative rather
than absolute and run both versions on the same idle host.

The vLLM 0.26 upgrade was measured with five repetitions on an 8-core Intel
Core Ultra 7 258V CPU:

| Stack | Engine initialization | First request | Steady-state median |
| --- | ---: | ---: | ---: |
| vLLM 0.8.4, OpenVINO 2026.2.1 | 24.49 s | 38.90 output tok/s | 89.29 output tok/s |
| vLLM 0.26.0, OpenVINO 2026.3.0 | 35.49 s | 67.68 output tok/s | 101.88 output tok/s |

The new worker initializes vLLM's compiled CPU sampler during engine startup.
This increases reported initialization time but prevents compilation from
delaying the first user request.

### GPU performance tips

GPU device implements the logic for automatic detection of available GPU memory and, by default, tries to reserve as much memory as possible for the KV cache (taking into account `gpu_memory_utilization` option). However, this behavior can be overridden by explicitly specifying the desired amount of memory for the KV cache using `VLLM_OPENVINO_KVCACHE_SPACE` environment variable (e.g, `VLLM_OPENVINO_KVCACHE_SPACE=8` means 8 GB space for KV cache).

Additionally, GPU device supports `VLLM_OPENVINO_KV_CACHE_PRECISION` (e.g. `i8` or `fp16`) to control KV cache precision (default value is device-specific).

Quantized weights can reduce GPU memory use. Both 8-bit and 4-bit weight
compression are supported for models exported with Optimum Intel.

Example GPU throughput command:

```console
VLLM_OPENVINO_DEVICE=GPU \
VLLM_OPENVINO_KV_CACHE_PRECISION=i8 \
VLLM_OPENVINO_ENABLE_QUANTIZED_WEIGHTS=ON \
vllm bench throughput \
    --model meta-llama/Llama-2-7b-chat-hf \
    --dataset-name random \
    --num-prompts 256 \
    --random-input-len 128 \
    --random-output-len 128
```

## Limitations

- LoRA serving is not supported.
- Prefix caching and asynchronous scheduling are not supported.
- Multimodal and encoder-decoder models are not supported.
- Tensor and pipeline parallelism are not supported.

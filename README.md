# Local

## uv

Use [uv](https://docs.astral.sh/uv/) package manager to manage the
installation of the plugin and its dependencies.

### Install uv

[Installing uv](https://docs.astral.sh/uv/getting-started/installation/)

### Install vLLM OpenVINO

```
git clone https://github.com/vllm-project/vllm-openvino.git
cd vllm-openvino
uv venv
VLLM_TARGET_DEVICE=empty uv pip install -e .
```
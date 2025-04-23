# OpenVINO plugin for vLLM

## Installation

### Local installation

#### Installing via uv

To manage the installation of the plugin and its dependencies,
utilize the [uv](https://docs.astral.sh/uv/) package manager:
[installing uv](https://docs.astral.sh/uv/getting-started/installation/)

### Install vLLM OpenVINO

```
git clone https://github.com/vllm-project/vllm-openvino.git
cd vllm-openvino
uv venv
VLLM_TARGET_DEVICE=empty uv pip install -e .
# test
source .venv/bin/activate
python examples/offline_inference_openvino.py
```
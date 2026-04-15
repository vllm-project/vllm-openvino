
# OpenVINO [CPU] plugin for vLLM 

## Tested for

- vLLM == 0.16.0
- OpenVINO == 2026.1.0
- optimum-intel[nncf] == 1.27.0

## Set up using Python

### Pre-built wheels

Currently, there are no pre-build wheels.

### Build wheel from source

1. Install Python and ensure you have the latest pip. For example, on Ubuntu 22.04, you can run:

```console
sudo apt-get update  -y
sudo apt-get install python3-pip
pip install --upgrade pip
```

2. clone vllm-openvino and install prerequisites for the vLLM OpenVINO backend installation:

```console
git clone https://github.com/vllm-project/vllm-openvino.git
cd vllm-openvino
git checkout vllm_v16_cpu_only_plugin
```

3. install vLLM with OpenVINO backend:

```console
pip install -v .
```

> [!NOTE]
> Only OpenVINO CPU plugin in supported in this branch 


## Extra information
## Models tested on ARM
✅ OPT

✅ LLAMA3

❌ GPTOSS
## Feature Support
Only Base Implementation is tested. 
Future testing and Support
- [x] Base Implementation
- [ ] Prefix caching (--enable-prefix-caching)
- [ ] Chunked prefill (--enable-chunked-prefill)
- [ ] Prefix caching + chunked prefill


## vLLM OpenVINO runtime configuration

You can pass the openvino compile config from vllm as additional config
``` config
### config.yaml
openvino:
  VLLM_OPENVINO_DEVICE: "CPU"
  VLLM_OPENVINO_KVCACHE_SPACE: 10
  VLLM_OPENVINO_ENABLE_QUANTIZED_WEIGHTS: true
  OPENVINO_RUNTIME_CONFIG:
    KV_CACHE_PRECISION: "u8"
    DYNAMIC_QUANTIZATION_GROUP_SIZE: "18446744073709551615"
``` 
vllm LLM API can be initialized with this additional config
``` code
import yaml
with open("config.yaml", "r") as file:
        data = yaml.safe_load(file)
llm = LLM(model="facebook/opt-125m", additional_config=data)
```

## Limitations

- LoRA serving is not supported.
- Only LLM models are currently supported. LLaVa and encoder-decoder models are not currently enabled in vLLM OpenVINO integration.
- Tensor and pipeline parallelism are not currently enabled in vLLM integration.

# The vLLM Dockerfile is used to construct vLLM image that can be directly used
# to run the OpenAI compatible server.

FROM ubuntu:22.04 AS dev

RUN apt-get update -y && \
    apt-get install -y git python3-pip
    
WORKDIR /workspace

COPY . .

RUN python3 -m pip install -U pip
# build vLLM with OpenVINO backend
RUN VLLM_TARGET_DEVICE="empty" PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu" python3 -m pip install /workspace
# In x86, triton will be installed by vllm. But in OpenVINO plugin, triton doesn't work correctly. we need to uninstall it.
RUN python3 -m pip uninstall -y triton
# copy samples
COPY examples/ /workspace/examples

CMD ["/bin/bash"]

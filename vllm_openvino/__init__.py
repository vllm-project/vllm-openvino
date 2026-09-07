import os
from logging.config import dictConfig

import vllm.envs as vllm_envs
from vllm.logger import DEFAULT_LOGGING_CONFIG

from vllm_openvino.envs import environment_variables


def register():
    """Register OpenVINO."""
    # Register plugin-owned variables before vLLM validates the environment.
    vllm_envs.environment_variables.update(environment_variables)
    # OpenVINO provides its own V1 model runner.
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    return "vllm_openvino.platform.OpenVinoPlatform"


def _init_logging():
    """Setup logging, extending from the vLLM logging config"""
    config = {**DEFAULT_LOGGING_CONFIG}

    # Copy the vLLM logging configurations
    config["formatters"]["vllm_openvino"] = DEFAULT_LOGGING_CONFIG["formatters"][
        "vllm"]

    handler_config = DEFAULT_LOGGING_CONFIG["handlers"]["vllm"]
    handler_config["formatter"] = "vllm_openvino"
    config["handlers"]["vllm_openvino"] = handler_config

    logger_config = DEFAULT_LOGGING_CONFIG["loggers"]["vllm"]
    logger_config["handlers"] = ["vllm_openvino"]
    config["loggers"]["vllm_openvino"] = logger_config

    dictConfig(config)


_init_logging()

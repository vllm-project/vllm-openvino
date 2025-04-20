from logging.config import dictConfig

from vllm.logger import DEFAULT_LOGGING_CONFIG


def register():
    """Register OpenVINO."""
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
from typing import ClassVar


class VersionDefaults:
    """Default framework versions to use when building LLM server images."""

    GUIDELLM = "72374ef"
    SGLANG = "v0.4.9.post6-cu126"
    TENSORRT_LLM = "1.1.0rc0"
    TOKASAURUS = "eecacdb"
    VLLM = "v0.10.0"

    LLM_SERVERS: ClassVar[dict[str, str]] = {
        "sglang": SGLANG,
        "tensorrt-llm": TENSORRT_LLM,
        "tokasaurus": TOKASAURUS,
        "vllm": VLLM,
        "vllm-pd-disaggregation": VLLM,
    }


SECONDS = 1
MINUTES = 60 * SECONDS
HOURS = 60 * MINUTES

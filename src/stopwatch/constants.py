from typing import ClassVar


class VersionDefaults:
    """Default framework versions to use when building LLM server images."""

    GUIDELLM = "1261fe8"
    SGLANG = "v0.4.10.post2-cu126"
    TENSORRT_LLM = "1.0.0rc4"
    TOKASAURUS = "0.0.3.post1"
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

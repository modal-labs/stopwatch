import typing


class VersionDefaults:
    """Default framework versions to use when building LLM server images."""

    GUIDELLM = "e3be0ef"
    SGLANG = "v0.4.6.post5-cu124"
    TENSORRT_LLM = "0.21.0rc0"
    TOKASAURUS = "e7f3c9d"
    VLLM = "v0.9.0"

    LLM_SERVERS: typing.ClassVar[dict[str, str]] = {
        "sglang": SGLANG,
        "tensorrt-llm": TENSORRT_LLM,
        "tokasaurus": TOKASAURUS,
        "vllm": VLLM,
    }


SECONDS = 1
MINUTES = 60 * SECONDS
HOURS = 60 * MINUTES

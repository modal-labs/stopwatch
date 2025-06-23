import typing


class VersionDefaults:
    """Default framework versions to use when building LLM server images."""

    GUIDELLM = "10ca1d4"
    SGLANG = "v0.4.7.post1-cu124"
    TENSORRT_LLM = "0.21.0rc2"
    TOKASAURUS = "c01e494"
    VLLM = "v0.9.1"

    LLM_SERVERS: typing.ClassVar[dict[str, str]] = {
        "sglang": SGLANG,
        "tensorrt-llm": TENSORRT_LLM,
        "tokasaurus": TOKASAURUS,
        "vllm": VLLM,
    }


SECONDS = 1
MINUTES = 60 * SECONDS
HOURS = 60 * MINUTES

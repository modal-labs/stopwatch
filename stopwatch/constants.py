import typing


class VersionDefaults:
    """Default framework versions to use when building LLM server images."""

    GUIDELLM = "0e78c65"
    SGLANG = "v0.4.8.post1-cu128-b200"
    TENSORRT_LLM = "1.0.0rc0"
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


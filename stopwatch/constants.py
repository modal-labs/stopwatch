class VersionDefaults:
    GUIDELLM = "e6f3dfc"
    SGLANG = "v0.4.6.post2-cu124"
    TENSORRT_LLM = "0.20.0rc1"
    VLLM = "v0.8.5.post1"

    LLM_SERVERS = {
        "sglang": SGLANG,
        "tensorrt-llm": TENSORRT_LLM,
        "vllm": VLLM,
    }


SECONDS = 1
MINUTES = 60 * SECONDS
HOURS = 60 * MINUTES

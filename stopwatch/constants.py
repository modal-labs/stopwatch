class VersionDefaults:
    GUIDELLM = "678adea"
    SGLANG = "v0.4.6.post5-cu124"
    TENSORRT_LLM = "0.21.0rc0"
    VLLM = "v0.9.0"

    LLM_SERVERS = {
        "sglang": SGLANG,
        "tensorrt-llm": TENSORRT_LLM,
        "vllm": VLLM,
    }


SECONDS = 1
MINUTES = 60 * SECONDS
HOURS = 60 * MINUTES

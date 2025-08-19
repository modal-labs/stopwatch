from enum import Enum
from typing import ClassVar

SECONDS = 1
MINUTES = 60 * SECONDS
HOURS = 60 * MINUTES

# Volume mount paths
DB_PATH = "/db"
HF_CACHE_PATH = "/cache"
TRACES_PATH = "/traces"
VLLM_CACHE_PATH = "/root/.cache/vllm"


class RateType(Enum):
    """Types of rates for running benchmarks."""

    constant = "constant"
    synchronous = "synchronous"
    throughput = "throughput"


class LLMServerType(Enum):
    """Types of LLM servers."""

    sglang = "sglang"
    tensorrt_llm = "tensorrt-llm"
    tokasaurus = "tokasaurus"
    vllm = "vllm"
    vllm_pd_disaggregation = "vllm-pd-disaggregation"


class VersionDefaults:
    """Default framework versions to use when building LLM server images."""

    GUIDELLM = "1261fe8"
    SGLANG = "v0.4.10.post2-cu126"
    TENSORRT_LLM = "1.0.0rc4"
    TOKASAURUS = "0.0.3.post1"
    VLLM = "v0.10.1"

    LLM_SERVERS: ClassVar[dict[LLMServerType, str]] = {
        LLMServerType.sglang: SGLANG,
        LLMServerType.tensorrt_llm: TENSORRT_LLM,
        LLMServerType.tokasaurus: TOKASAURUS,
        LLMServerType.vllm: VLLM,
        LLMServerType.vllm_pd_disaggregation: VLLM,
    }

from enum import Enum

SECONDS = 1
MINUTES = 60 * SECONDS
HOURS = 60 * MINUTES

# Volume mount paths
DB_PATH = "/db"
HF_CACHE_PATH = "/cache"
TRACES_PATH = "/traces"
VLLM_CACHE_PATH = "/root/.cache/vllm"

GUIDELLM_VERSION = "1261fe8"
TENSORRT_LLM_CUDA_VERSION = "12.9.1"
TOKASAURUS_CUDA_VERSION = "12.4.1"


class RateType(str, Enum):
    """Types of rates for running benchmarks."""

    constant = "constant"
    synchronous = "synchronous"
    throughput = "throughput"


class LLMServerType(str, Enum):
    """Types of LLM servers."""

    sglang = "sglang"
    tensorrt_llm = "tensorrt-llm"
    tokasaurus = "tokasaurus"
    vllm = "vllm"
    vllm_pd_disaggregation = "vllm-pd-disaggregation"

    def get_version(self) -> str:
        """Get the latest version of the LLM server."""

        versions = {
            LLMServerType.sglang: "v0.4.10.post2-cu126",
            LLMServerType.tensorrt_llm: "1.0.0rc4",
            LLMServerType.tokasaurus: "0.0.3.post1",
            LLMServerType.vllm: "v0.10.1",
            LLMServerType.vllm_pd_disaggregation: "v0.10.1",
        }

        return versions[self]

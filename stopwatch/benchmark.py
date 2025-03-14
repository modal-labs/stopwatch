from typing import Dict, List


class BenchmarkDefaults:
    DATA = "prompt_tokens=512,generated_tokens=128"
    DATA_TYPE = "emulated"
    GPU = "H100"
    REGION = "us-ashburn-1"
    LLM_ENGINE_TYPE = "vllm"
    LLM_ENGINE_VERSION = "0.7.3"
    LLM_ENV_VARS = {}
    LLM_EXTRA_ARGS = []


def get_benchmark_fingerprint(
    model: str,
    data: str = BenchmarkDefaults.DATA,
    data_type: str = BenchmarkDefaults.DATA_TYPE,
    gpu: str = BenchmarkDefaults.GPU,
    region: str = BenchmarkDefaults.REGION,
    llm_engine_type: str = BenchmarkDefaults.LLM_ENGINE_TYPE,
    llm_engine_version: BenchmarkDefaults.LLM_ENGINE_VERSION, 
    llm_env_vars: Dict[str, str] = BenchmarkDefaults.LLM_ENV_VARS,
    llm_extra_args: List[str] = BenchmarkDefaults.LLM_EXTRA_ARGS,
    repeat_index: int = 0,
):
    env_vars = "-".join([f"{k}={v}" for k, v in sorted(vllm_env_vars.items())])
    fingerprint = f"{model}-{data}-{data_type}-{gpu}-{region}-{vllm_docker_tag}-{env_vars}-{vllm_extra_args}"
    if repeat_index > 0:
        fingerprint += f"-{repeat_index}"
    return fingerprint

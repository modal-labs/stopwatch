from typing import Dict, List


class BenchmarkDefaults:
    DATA = "prompt_tokens=512,generated_tokens=128"
    DATA_TYPE = "emulated"
    GPU = "H100"
    REGION = "us-ashburn-1"
    VLLM_DOCKER_TAG = "v0.7.3"
    VLLM_ENV_VARS = {}
    VLLM_EXTRA_ARGS = []


def get_benchmark_fingerprint(
    model: str,
    data: str = BenchmarkDefaults.DATA,
    data_type: str = BenchmarkDefaults.DATA_TYPE,
    gpu: str = BenchmarkDefaults.GPU,
    region: str = BenchmarkDefaults.REGION,
    vllm_docker_tag: str = BenchmarkDefaults.VLLM_DOCKER_TAG,
    vllm_env_vars: Dict[str, str] = BenchmarkDefaults.VLLM_ENV_VARS,
    vllm_extra_args: List[str] = BenchmarkDefaults.VLLM_EXTRA_ARGS,
    repeat_index: int = 0,
):
    env_vars = "-".join([f"{k}={v}" for k, v in sorted(vllm_env_vars.items())])
    fingerprint = f"{model}-{data}-{data_type}-{gpu}-{region}-{vllm_docker_tag}-{env_vars}-{vllm_extra_args}"
    if repeat_index > 0:
        fingerprint += f"-{repeat_index}"
    return fingerprint

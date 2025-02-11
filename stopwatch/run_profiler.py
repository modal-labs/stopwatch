import modal

from .benchmark import BenchmarkDefaults
from .resources import app
from .vllm_runner import vllm


TIMEOUT = 60 * 60  # 1 hour
TRACES_PATH = "/traces"


profiling_image = modal.Image.debian_slim().pip_install("openai")

with profiling_image.imports():
    from typing import Dict, List
    from openai import OpenAI


@app.function(
    image=profiling_image,
    timeout=TIMEOUT,
    cloud="oci",
    region="us-chicago-1",
)
def run_profiler(
    model: str,
    num_requests: int = 10,
    gpu: str = BenchmarkDefaults.GPU,
    vllm_docker_tag: str = BenchmarkDefaults.VLLM_DOCKER_TAG,
    vllm_env_vars: Dict[str, str] = BenchmarkDefaults.VLLM_ENV_VARS,
    vllm_extra_args: List[str] = BenchmarkDefaults.VLLM_EXTRA_ARGS,
):
    """
    Runs the torch profiler on a model.

    Args:
        model (str): The name of the model to profile.
    """

    print(f"Starting profiler with {model}")

    # Start vLLM server in background
    with vllm(
        model,
        docker_tag=vllm_docker_tag,
        env_vars={
            **vllm_env_vars,
            "VLLM_TORCH_PROFILER_DIR": TRACES_PATH,
            "VLLM_RPC_TIMEOUT": "1800000",
        },
        extra_args=vllm_extra_args,
        gpu=gpu,
        profile=True,
    ) as vllm_url:
        client = OpenAI(api_key="EMPTY", base_url=f"{vllm_url}/v1")

        models = client.models.list()
        model = models.data[0].id

        for _ in range(num_requests):
            client.completions.create(
                model=model,
                prompt="A robot may not injure a human being",
                echo=False,
                n=2,
                stream=False,
                logprobs=3,
            )

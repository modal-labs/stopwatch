import modal

from .benchmark import BenchmarkDefaults
from .resources import app, hf_secret, traces_volume
from .vllm_runner import vllm


TIMEOUT = 60 * 60  # 1 hour
TRACES_PATH = "/traces"


profiling_image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install("git+https://github.com/neuralmagic/guidellm.git", "openai")
)

with profiling_image.imports():
    from typing import Dict, List

    from guidellm.request import EmulatedRequestGenerator
    from openai import OpenAI


@app.function(
    image=profiling_image,
    secrets=[hf_secret],
    volumes={TRACES_PATH: traces_volume},
    timeout=TIMEOUT,
    region="us-ashburn-1",
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

    vllm_env_vars["VLLM_TORCH_PROFILER_DIR"] = TRACES_PATH
    vllm_env_vars["VLLM_RPC_TIMEOUT"] = "1800000"

    extra_query = {
        "model": model,
    }

    if len(vllm_extra_args) > 0:
        extra_query["extra_vllm_args"] = " ".join(vllm_extra_args)
    if len(vllm_env_vars) > 0:
        extra_query["vllm_env_vars"] = " ".join(
            f"{k}={v}" for k, v in vllm_env_vars.items()
        )

    # Start vLLM server in background
    with vllm(
        docker_tag=vllm_docker_tag,
        extra_query=extra_query,
        gpu=gpu,
        profile=True,
    ) as vllm_url:
        client = OpenAI(api_key="EMPTY", base_url=f"{vllm_url}/v1")
        erg = EmulatedRequestGenerator("", tokenizer=model)

        for _ in range(num_requests):
            prompt = erg.sample_prompt(512)
            client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens=1,
                echo=False,
                stream=False,
                extra_query=extra_query,
            )

    # Find and return trace path
    most_recent_path = None
    most_recent_timestamp = 0

    for file in traces_volume.iterdir("/"):
        if file.mtime > most_recent_timestamp:
            most_recent_timestamp = file.mtime
            most_recent_path = file.path

    return most_recent_path

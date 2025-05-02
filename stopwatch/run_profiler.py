import modal

from .resources import app, hf_secret, traces_volume
from .vllm_runner import vllm


TIMEOUT = 60 * 60  # 1 hour
TRACES_PATH = "/traces"


profiling_image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install("git+https://github.com/jackcook/guidellm.git@25dcc2e", "openai")
    .add_local_python_source("cli")
)

with profiling_image.imports():
    from typing import Any, Mapping, Optional

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
    llm_server_type: str,
    gpu: str,
    num_requests: int = 10,
    llm_server_config: Optional[Mapping[str, Any]] = None,
):
    """
    Runs the torch profiler on a model.

    Args:
        model (str): The name of the model to profile.
    """

    print(f"Starting profiler with {model}")

    assert llm_server_type == "vllm", "Profiling is only supported with vLLM"

    if "env_vars" not in llm_server_config:
        llm_server_config["env_vars"] = {}

    llm_server_config["env_vars"]["VLLM_TORCH_PROFILER_DIR"] = TRACES_PATH
    llm_server_config["env_vars"]["VLLM_RPC_TIMEOUT"] = "1800000"

    # Start vLLM server in background
    with vllm(
        model=model,
        llm_server_config=llm_server_config,
        gpu=gpu,
        profile=True,
    ) as (vllm_url, extra_query):
        client = OpenAI(api_key="EMPTY", base_url=f"{vllm_url}/v1")
        erg = EmulatedRequestGenerator("", tokenizer=model)

        for _ in range(num_requests):
            prompt = erg.sample_prompt(512)
            client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens=8,
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

import logging
import time

import modal

from .constants import VersionDefaults
from .llm_server import llm_server
from .resources import app, hf_secret, traces_volume

TIMEOUT = 60 * 60  # 1 hour
TRACES_PATH = "/traces"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


profiling_image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        f"git+https://github.com/jackcook/guidellm.git#{VersionDefaults.GUIDELLM}",
        "openai",
    )
    .add_local_python_source("cli")
)

with profiling_image.imports():
    from collections.abc import Mapping
    from typing import Any

    from guidellm.dataset import SyntheticDatasetConfig, SyntheticTextItemsGenerator
    from openai import OpenAI
    from transformers import AutoTokenizer


@app.function(
    image=profiling_image,
    secrets=[hf_secret],
    volumes={TRACES_PATH: traces_volume},
    timeout=TIMEOUT,
)
def run_profiler(
    llm_server_type: str,
    model: str,
    gpu: str = "H100",
    server_region: str = "us-chicago-1",
    num_requests: int = 10,
    prompt_tokens: int = 512,
    output_tokens: int = 8,
    llm_server_config: Mapping[str, Any] | None = None,
) -> str:
    """
    Run the PyTorch profiler alongside an LLM server. Currently, only vLLM is
    supported.

    :param: llm_server_type: The type of LLM server to use. Currently, only "vLLM" is
        supported.
    :param: model: The model to use.
    :param: gpu: The GPU to use.
    :param: server_region: The region in which to run the server.
    :param: num_requests: The number of requests to make. Traces get large very
        quickly, so this should be kept small.
    :param: prompt_tokens: The number of tokens to include in each request's prompt.
    :param: output_tokens: The number of tokens to generate in each request.
    :param: llm_server_config: The configuration for the LLM server.
    :return: The path to the trace file.
    """

    logger.info("Starting profiler with %s", model)

    if llm_server_type != "vllm":
        msg = "Profiling is only supported with vLLM"
        raise ValueError(msg)

    if llm_server_config is None:
        llm_server_config = {}

    if "env_vars" not in llm_server_config:
        llm_server_config["env_vars"] = {}

    llm_server_config["env_vars"]["VLLM_TORCH_PROFILER_DIR"] = TRACES_PATH
    llm_server_config["env_vars"]["VLLM_RPC_TIMEOUT"] = "1800000"

    generator_config = SyntheticDatasetConfig(
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
    )
    tokenizer = AutoTokenizer.from_pretrained(model)
    text_generator = iter(
        SyntheticTextItemsGenerator(
            config=generator_config,
            processor=tokenizer,
            random_seed=42,
        ),
    )

    # Start vLLM server in background
    with llm_server(
        "vllm",
        model=model,
        gpu=gpu,
        region=server_region,
        server_config=llm_server_config,
        profile=True,
    ) as (vllm_url, extra_query):
        client = OpenAI(api_key="EMPTY", base_url=f"{vllm_url}/v1")

        for _ in range(num_requests):
            client.completions.create(
                model=model,
                prompt=next(text_generator)["prompt"],
                max_tokens=output_tokens,
                echo=False,
                stream=False,
                extra_query=extra_query,
            )

    # Find and return trace path
    most_recent_path = None
    most_recent_size = 0
    most_recent_timestamp = 0

    for file in traces_volume.iterdir("/"):
        if file.mtime > most_recent_timestamp:
            most_recent_path = file.path
            most_recent_size = file.size
            most_recent_timestamp = file.mtime

    # Wait for profiler to finish writing profiling output before returning
    while True:
        time.sleep(5)

        traces_volume.reload()

        for file in traces_volume.iterdir("/"):
            if file.path == most_recent_path:
                latest_size = file.size
                break

        if latest_size == most_recent_size:
            break

        most_recent_size = latest_size

    return most_recent_path

import logging
import time

import modal

from .constants import VersionDefaults
from .resources import app, hf_secret, traces_volume

TIMEOUT = 60 * 60  # 1 hour
TRACES_PATH = "/traces"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


profiling_image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .uv_pip_install(
        f"git+https://github.com/neuralmagic/guidellm.git#{VersionDefaults.GUIDELLM}",
        "openai",
    )
)

with profiling_image.imports():
    import requests
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
    endpoint: str,
    model: str,
    num_requests: int = 10,
    prompt_tokens: int = 512,
    output_tokens: int = 8,
) -> str:
    """
    Run the PyTorch profiler alongside an LLM server. Currently, only vLLM is
    supported.

    :param: endpoint: The endpoint of the OpenAI-compatible LLM server to use.
    :param: model: The model to use.
    :param: num_requests: The number of requests to make. Traces get large very
        quickly, so this should be kept small.
    :param: prompt_tokens: The number of tokens to include in each request's prompt.
    :param: output_tokens: The number of tokens to generate in each request.
    :return: The path to the trace file.
    """

    logger.info("Starting profiler with %s", model)

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

    # Start profiler
    requests.post(f"{endpoint}/start_profile")

    # Start vLLM server in background
    client = OpenAI(api_key="EMPTY", base_url=f"{endpoint}/v1")

    for _ in range(num_requests):
        client.completions.create(
            model=model,
            prompt=next(text_generator)["prompt"],
            max_tokens=output_tokens,
            echo=False,
            stream=False,
        )

    # Stop profiler
    requests.post(f"{endpoint}/stop_profile")

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

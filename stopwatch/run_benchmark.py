import logging
import os
import time

import modal

from .resources import app, hf_secret, results_volume, tunnel_urls
from .vllm_runner import get_vllm_cls


CONTAINER_IDLE_TIMEOUT = 5  # 5 seconds
MAX_SECONDS_PER_BENCHMARK_RUN = 120
RESULTS_PATH = "/results"
TIMEOUT = 60 * 60  # 1 hour

benchmarking_image = modal.Image.debian_slim().pip_install("guidellm", "requests")

with benchmarking_image.imports():
    from guidellm import generate_benchmark_report
    import requests


@app.function(
    image=benchmarking_image,
    secrets=[hf_secret],
    volumes={RESULTS_PATH: results_volume},
    cpu=4,
    memory=2048,
    container_idle_timeout=CONTAINER_IDLE_TIMEOUT,
    timeout=TIMEOUT,
    cloud="oci",
    region="us-chicago-1",
)
def run_benchmark(
    model: str,
    data: str = "prompt_tokens=512,generated_tokens=128",
    gpu: str = "H100",
    vllm_docker_tag: str = "latest",
    vllm_env_vars: dict = {},
    vllm_extra_args: list = [],
):
    """Benchmarks a model on Modal.

    Args:
        model (str): Name of the model to benchmark.
        data (str, optional): The data source to use for benchmarking.
            Depending on the data-type, it should be a path to a data file
            containing prompts to run (ex: data.txt), a HuggingFace dataset
            name (ex: 'neuralmagic/LLM_compression_calibration'), or a
            configuration for emulated data (ex:
            'prompt_tokens=128,generated_tokens=128').
    """

    logging.info(f"Starting benchmark for {model}")
    fc_id = modal.current_function_call_id()

    # Start vLLM server in background
    vLLM = get_vllm_cls(docker_tag=vllm_docker_tag, gpu=gpu)
    vllm = vLLM()
    vllm_fc = vllm.start.spawn(
        caller_id=fc_id,
        env_vars=vllm_env_vars,
        vllm_args=["--model", model, *vllm_extra_args],
    )

    # Wait for vLLM server to start
    while True:
        time.sleep(5)
        url = tunnel_urls.get(fc_id, None)

        if url is None:
            continue

        try:
            requests.get(url)
            logging.info(f"Connected to vLLM instance at {url}")
            break
        except Exception:
            continue

    # Run benchmark with guidellm
    generate_benchmark_report(
        target=url,
        backend="openai_server",
        model=model,
        data=data,
        data_type="emulated",
        tokenizer=None,
        rate_type="sweep",
        rate=None,
        max_seconds=MAX_SECONDS_PER_BENCHMARK_RUN,
        max_requests=None,
        output_path=os.path.join(RESULTS_PATH, f"{fc_id}.json"),
        cont_refresh_table=False,
    )

    logging.info("Benchmark complete")
    vllm_fc.cancel(terminate_containers=True)

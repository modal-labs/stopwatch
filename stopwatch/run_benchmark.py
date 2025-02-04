import logging
import os
import time

import modal

from .resources import app, results_volume, tunnel_urls
from .vllm_runner import vLLM


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
    container_idle_timeout=CONTAINER_IDLE_TIMEOUT,
    timeout=TIMEOUT,
    volumes={RESULTS_PATH: results_volume},
)
def run_benchmark(model: str, data: str = "neuralmagic/LLM_compression_calibration"):
    """Benchmarks a model on Modal.

    Args:
        model (str): Name of the model to benchmark.
        data (str, optional): Name of the dataset to use for benchmarking.
            Defaults to "neuralmagic/LLM_compression_calibration".
    """

    fc_id = modal.current_function_call_id()

    # Start vLLM server in background
    vllm = vLLM()
    vllm.start.spawn(model=model, caller_id=fc_id)

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
        data_type="transformers",
        tokenizer=None,
        rate_type="sweep",
        rate=None,
        max_seconds=MAX_SECONDS_PER_BENCHMARK_RUN,
        max_requests=None,
        output_path=os.path.join(RESULTS_PATH, f"{fc_id}.json"),
        cont_refresh_table=False,
    )

    logging.info("Benchmark complete")

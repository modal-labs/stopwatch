import modal

from .benchmark import BenchmarkDefaults, get_benchmark_fingerprint
from .resources import app, hf_secret, results_dict, results_volume, tunnel_urls
from .vllm_runner import get_vllm_cls


CONTAINER_IDLE_TIMEOUT = 5  # 5 seconds
MAX_SECONDS_PER_BENCHMARK_RUN = 120  # 2 minutes
RESULTS_PATH = "/results"
TIMEOUT = 60 * 60  # 1 hour

benchmarking_image = modal.Image.debian_slim().pip_install(
    "guidellm", "prometheus-client", "requests"
)

with benchmarking_image.imports():
    from typing import Dict, List
    import os
    import time

    from guidellm import generate_benchmark_report
    import requests

    from .custom_metrics import vllm_monkey_patch


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
    data: str = BenchmarkDefaults.DATA,
    data_type: str = BenchmarkDefaults.DATA_TYPE,
    gpu: str = BenchmarkDefaults.GPU,
    vllm_docker_tag: str = BenchmarkDefaults.VLLM_DOCKER_TAG,
    vllm_env_vars: Dict[str, str] = BenchmarkDefaults.VLLM_ENV_VARS,
    vllm_extra_args: List[str] = BenchmarkDefaults.VLLM_EXTRA_ARGS,
):
    """Benchmarks a model on Modal.

    Args:
        model (str, required): Name of the model to benchmark.
        data (str): The data source to use for benchmarking. Depending on the
            data-type, it should be a path to a data file containing prompts to
            run (ex: data.txt), a HuggingFace dataset name (ex:
            'neuralmagic/LLM_compression_calibration'), or a configuration for
            emulated data (ex: 'prompt_tokens=128,generated_tokens=128').
        data_type (str): The type of data to use, such as 'emulated', 'file',
            or 'transformers'.
        gpu (str): The GPU to use for benchmarking.
        vllm_docker_tag (str): Tag of the vLLM server docker image. Defaults to
            `latest`.
        vllm_env_vars (dict): Environment variables to pass to the vLLM server.
        vllm_extra_args (list): Extra arguments to pass to the vLLM server.
    """

    print(f"Starting benchmark for {model}")
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
            print(f"Connected to vLLM instance at {url}")
            break
        except Exception:
            continue

    metrics_url = f"{url}/metrics"
    vllm_monkey_patch(metrics_url)

    # Run benchmark with guidellm
    generate_benchmark_report(
        target=f"{url}/v1",
        backend="openai_server",
        model=model,
        data=data,
        data_type=data_type,
        tokenizer=None,
        rate_type="sweep",
        rate=None,
        max_seconds=MAX_SECONDS_PER_BENCHMARK_RUN,
        max_requests=None,
        output_path=os.path.join(RESULTS_PATH, f"{fc_id}.json"),
        cont_refresh_table=False,
    )
    results_volume.commit()

    # Cache results path
    fingerprint = get_benchmark_fingerprint(
        model=model,
        data=data,
        data_type=data_type,
        gpu=gpu,
        vllm_docker_tag=vllm_docker_tag,
        vllm_env_vars=vllm_env_vars,
        vllm_extra_args=vllm_extra_args,
    )
    results_dict[fingerprint] = fc_id

    # Clean up
    print("Benchmark complete")
    vllm_fc.cancel(terminate_containers=True)

import modal

from .resources import app, hf_secret, results_volume
from .llm_server import llm_server


LLM_SERVER_TYPES = ["vllm", "sglang", "tensorrt-llm"]
MAX_SECONDS_PER_BENCHMARK = 120  # 2 minutes
RESULTS_PATH = "/results"
SCALEDOWN_WINDOW = 5  # 5 seconds
TIMEOUT = 30 * 60  # 30 minutes

benchmarking_image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "git+https://github.com/jackcook/guidellm.git#1eb26a9",
        "prometheus-client",
        "tiktoken",
    )
    .add_local_python_source("cli")
)

with benchmarking_image.imports():
    from typing import Any, Mapping, Optional
    import urllib.parse

    from guidellm.backend import Backend
    from guidellm.benchmark.benchmarker import GenerativeBenchmarker
    from guidellm.benchmark.profile import create_profile
    from guidellm.request import GenerativeRequestLoader

    from .custom_metrics.vllm import GenerativeBenchmarkerWithvLLMMetrics


def benchmark_runner_cls(region: str):
    def decorator(cls):
        return app.cls(
            image=benchmarking_image,
            secrets=[hf_secret],
            volumes={RESULTS_PATH: results_volume},
            cpu=4,
            memory=2048,
            scaledown_window=SCALEDOWN_WINDOW,
            timeout=TIMEOUT,
            region=region,
        )(cls)

    return decorator


class BenchmarkRunner:
    @modal.method()
    async def run_benchmark(
        self,
        llm_server_type: str,
        model: str,
        rate_type: str,
        data: str,
        gpu: str,
        server_region: str,
        llm_server_config: Optional[Mapping[str, Any]] = None,
        rate: Optional[float] = None,
        **kwargs,
    ):
        """Benchmarks a LLM deployment on Modal.

        Args:
            llm_server_type (str): The server to use for benchmarking, either
                'vllm', 'sglang', or 'tensorrt-llm'.
            model (str): Name of the model to benchmark.
            rate_type (str): The type of rate to use for benchmarking, either
                'constant', 'synchronous', or 'throughput'.
            data (str): A configuration for emulated data (e.g.:
                'prompt_tokens=128,output_tokens=128').
            gpu (str): The GPU to use for benchmarking.
            server_region (str): Region to run the LLM server on.
            llm_server_config (dict): Configuration for the LLM server.
            rate (float): If rate_type is 'constant', optionally specify the
                number of requests that should be made per second.
        """

        if llm_server_type not in LLM_SERVER_TYPES:
            raise ValueError(
                f"Invalid value for llm_server: {llm_server_type}. Must be one of {LLM_SERVER_TYPES}"
            )
        elif llm_server_config is None:
            llm_server_config = {}

        # Start LLM server in background
        with llm_server(
            llm_server_type,
            model=model,
            gpu=gpu,
            region=server_region,
            server_config=llm_server_config,
        ) as (llm_server_url, extra_query):
            extra_query_args = urllib.parse.urlencode(extra_query)
            metrics_url = f"{llm_server_url}/metrics?{extra_query_args}"

            # Create backend
            backend = Backend.create(
                "openai_http",
                target=f"{llm_server_url}/v1",
                model=model,
                extra_query=extra_query,
            )
            await backend.validate()

            print(f"Connected to backend for model {backend.model}.")
            processor = llm_server_config.get("tokenizer", model)

            request_loader = GenerativeRequestLoader(
                data=data,
                data_args=None,
                processor=processor,
                processor_args=None,
                shuffle=False,
                iter_type="infinite",
                random_seed=42,
            )
            unique_requests = request_loader.num_unique_items(raise_err=False)
            print(
                f"Created loader with {unique_requests} unique requests from {data}.\n\n"
                if unique_requests > 0
                else f"Created loader with unknown number unique requests from {data}.\n\n"
            )

            profile = create_profile(rate_type=rate_type, rate=rate)
            benchmarker_kwargs = {
                "backend": backend,
                "request_loader": request_loader,
                "request_loader_description": request_loader.description,
                "benchmark_save_extras": None,
                "processor": processor,
                "processor_args": None,
            }

            if llm_server_type == "vllm":
                benchmarker = GenerativeBenchmarkerWithvLLMMetrics(
                    vllm_metrics_url=metrics_url,
                    **benchmarker_kwargs,
                )
            else:
                benchmarker = GenerativeBenchmarker(**benchmarker_kwargs)

            async for result in benchmarker.run(
                profile=profile,
                max_number_per_strategy=None,
                max_duration_per_strategy=MAX_SECONDS_PER_BENCHMARK,
                warmup_percent_per_strategy=None,
                cooldown_percent_per_strategy=None,
            ):
                if result.type_ == "benchmark_compiled":
                    if result.current_benchmark is None:
                        raise ValueError("Current benchmark is None")
                    return result.current_benchmark.model_dump()


@benchmark_runner_cls(region="us-ashburn-1")
class BenchmarkRunner_OCI_USASHBURN1(BenchmarkRunner):
    pass


@benchmark_runner_cls(region="us-chicago-1")
class BenchmarkRunner_OCI_USCHICAGO1(BenchmarkRunner):
    pass


@benchmark_runner_cls(region="us-east-1")
class BenchmarkRunner_AWS_USEAST1(BenchmarkRunner):
    pass


@benchmark_runner_cls(region="us-east-2")
class BenchmarkRunner_AWS_USEAST2(BenchmarkRunner):
    pass


@benchmark_runner_cls(region="us-east4")
class BenchmarkRunner_GCP_USEAST4(BenchmarkRunner):
    pass


all_benchmark_runner_classes = {
    "us-ashburn-1": BenchmarkRunner_OCI_USASHBURN1,
    "us-east-1": BenchmarkRunner_AWS_USEAST1,
    "us-east-2": BenchmarkRunner_AWS_USEAST2,
    "us-east4": BenchmarkRunner_GCP_USEAST4,
    "us-chicago-1": BenchmarkRunner_OCI_USCHICAGO1,
}

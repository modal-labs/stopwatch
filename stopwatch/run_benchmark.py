import modal

from .constants import VersionDefaults
from .resources import app, hf_secret, results_volume
from .llm_server import llm_server


LLM_SERVER_TYPES = ["vllm", "sglang", "tensorrt-llm"]
NUM_CORES = 2
RESULTS_PATH = "/results"
SCALEDOWN_WINDOW = 5  # 5 seconds
TIMEOUT = 4 * 60 * 60  # 4 hours

benchmarking_image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        f"git+https://github.com/neuralmagic/guidellm.git#{VersionDefaults.GUIDELLM}",
        "prometheus-client",
        "tiktoken",
    )
    .env(
        {
            "GUIDELLM__MAX_WORKER_PROCESSES": f"{NUM_CORES - 1}",
        }
    )
    .add_local_python_source("cli")
)

with benchmarking_image.imports():
    from typing import Any, Mapping, Optional
    import urllib.parse

    from guidellm.backend import OpenAIHTTPBackend
    from guidellm.benchmark.benchmarker import GenerativeBenchmarker
    from guidellm.benchmark.profile import create_profile
    from guidellm.request import GenerativeRequestLoader

    from .custom_metrics.vllm import GenerativeBenchmarkerWithvLLMMetrics

    class CustomGenerativeRequestLoader(GenerativeRequestLoader):
        """
        A wrapper around GenerativeRequestLoader that allows for modifications to
        be made to GuideLLM requests.

        These are both useful when testing structured outputs, e.g.
        https://docs.vllm.ai/en/latest/features/structured_outputs.html
        """

        def __init__(
            self,
            extra_body: Optional[dict[str, Any]] = None,
            use_chat_completions: bool = False,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self.extra_body = extra_body or {}
            self.use_chat_completions = use_chat_completions

        def __iter__(self):
            for item in super().__iter__():
                for k, v in self.extra_body.items():
                    item.params[k] = v

                if self.use_chat_completions:
                    item.request_type = "chat_completions"

                yield item

        def __len__(self):
            return super().__len__()

    class CustomOpenAIHTTPBackend(OpenAIHTTPBackend):
        def _completions_payload(
            self,
            body: Optional[dict],
            orig_kwargs: Optional[dict],
            max_output_tokens: Optional[int],
            **kwargs,
        ) -> dict:
            payload = super()._completions_payload(
                body, orig_kwargs, max_output_tokens, **kwargs
            )

            if "max_completion_tokens" in payload:
                # The TensorRT-LLM server returns a 400 error if this is set.
                del payload["max_completion_tokens"]

            return payload


def benchmark_runner_cls(region: str):
    def decorator(cls):
        return app.cls(
            image=benchmarking_image,
            secrets=[hf_secret],
            volumes={RESULTS_PATH: results_volume},
            cpu=NUM_CORES,
            memory=1 * 1024,
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
        duration: Optional[float] = 120,  # 2 minutes
        llm_server_config: Optional[Mapping[str, Any]] = None,
        client_config: Optional[Mapping[str, Any]] = None,
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
            duration (float): The duration of the benchmark in seconds.
            llm_server_config (dict): Configuration for the LLM server.
            client_config (dict): Configuration for the GuideLLM client.
            rate (float): If rate_type is 'constant', optionally specify the
                number of requests that should be made per second.
        """

        if llm_server_type not in LLM_SERVER_TYPES:
            raise ValueError(
                f"Invalid value for llm_server: {llm_server_type}. Must be one of {LLM_SERVER_TYPES}"
            )

        if llm_server_config is None:
            llm_server_config = {}

        if client_config is None:
            client_config = {}

        # Create the request loader before starting the LLM server, since this can take
        # a long time for data configs with many prompt tokens.
        processor = llm_server_config.get("tokenizer", model)
        request_loader = CustomGenerativeRequestLoader(
            data=data,
            data_args=None,
            processor=processor,
            processor_args=None,
            shuffle=False,
            iter_type="infinite",
            random_seed=42,
            extra_body=client_config.get("extra_body", {}),
            use_chat_completions=client_config.get("use_chat_completions", False),
        )
        unique_requests = request_loader.num_unique_items(raise_err=False)
        print(
            f"Created loader with {unique_requests} unique requests from {data}.\n\n"
            if unique_requests > 0
            else f"Created loader with unknown number unique requests from {data}.\n\n"
        )

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
            backend = CustomOpenAIHTTPBackend(
                target=f"{llm_server_url}/v1",
                extra_query=extra_query,
            )
            await backend.validate()
            print(f"Connected to backend for model {backend.model}.")

            profile = create_profile(rate_type=rate_type, rate=rate)
            benchmarker_kwargs = {
                "backend": backend,
                "request_loader": request_loader,
                "request_loader_description": request_loader.description,
                "benchmark_save_extras": None,
                "processor": processor,
                "processor_args": None,
            }

            if llm_server_type == "vllm" and client_config.get(
                "collect_metrics", False
            ):
                benchmarker = GenerativeBenchmarkerWithvLLMMetrics(
                    vllm_metrics_url=metrics_url,
                    **benchmarker_kwargs,
                )
            else:
                benchmarker = GenerativeBenchmarker(**benchmarker_kwargs)

            async for result in benchmarker.run(
                profile=profile,
                max_number_per_strategy=None,
                max_duration_per_strategy=duration,
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

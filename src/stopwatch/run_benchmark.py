import asyncio
import itertools
import logging
import urllib.parse
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import modal

from .constants import HOURS, SECONDS, VersionDefaults
from .llm_servers import llm_server
from .resources import app, hf_secret, results_volume

# Delay between benchmarks when running multiple constant-rate benchmarks on the same
# LLM server. Must be less than the LLM server's scaledown_window.
DELAY_BETWEEN_BENCHMARKS = 15 * SECONDS

LLM_SERVER_TYPES = [
    "vllm",
    "vllm-pd-disaggregation",
    "sglang",
    "tensorrt-llm",
    "tokasaurus",
]
NUM_CORES = 2
RESULTS_PATH = "/results"
SCALEDOWN_WINDOW = 5 * SECONDS
TIMEOUT = 4 * HOURS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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
        },
    )
)

with benchmarking_image.imports():
    from guidellm.backend import OpenAIHTTPBackend
    from guidellm.backend.openai import EndpointType
    from guidellm.benchmark.benchmarker import GenerativeBenchmarker
    from guidellm.benchmark.profile import create_profile
    from guidellm.request import GenerationRequest, GenerativeRequestLoader

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
            extra_body: dict[str, Any] | None = None,
            *,
            use_chat_completions: bool = False,
            **kwargs: dict[str, Any],
        ) -> None:
            """
            Create a custom generative request loader.

            :param: extra_body: Extra parameters to add to the body of each request.
            :param: use_chat_completions: Whether to use the chat completions endpoint,
                as opposed to the text completions endpoint.
            :param: kwargs: Additional keyword arguments to pass to the
                GenerativeRequestLoader constructor.
            """

            super().__init__(**kwargs)
            self.extra_body = extra_body or {}
            self.use_chat_completions = use_chat_completions

        def __iter__(self) -> Iterator[GenerationRequest]:
            """Iterate over the requests in the loader."""

            for item in super().__iter__():
                for k, v in self.extra_body.items():
                    item.params[k] = v

                if self.use_chat_completions:
                    item.request_type = "chat_completions"

                yield item

        def __len__(self) -> int:
            """Return the number of unique requests in the loader."""
            return super().__len__()

    class CustomOpenAIHTTPBackend(OpenAIHTTPBackend):
        def _completions_payload(
            self,
            endpoint_type: EndpointType,
            orig_kwargs: dict | None,
            max_output_tokens: int | None,
            **kwargs: dict[str, Any],
        ) -> dict:
            payload = super()._completions_payload(
                endpoint_type,
                orig_kwargs,
                max_output_tokens,
                **kwargs,
            )

            if "max_completion_tokens" in payload:
                # The TensorRT-LLM server returns a 400 error if this is set.
                del payload["max_completion_tokens"]

            return payload


def benchmark_runner_cls(region: str) -> Callable:
    """
    Create a guidellm benchmark runner class that runs on Modal.

    :param: region: The region to run the benchmark in.
    """

    def decorator(cls: type) -> Callable:
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
        rate_type: str | list[str],
        data: str,
        gpu: str,
        server_region: str,
        duration: float | None = 120,  # 2 minutes
        llm_server_config: Mapping[str, Any] | None = None,
        client_config: Mapping[str, Any] | None = None,
        rate: float | list[float] | None = None,
        **kwargs: dict[str, Any],  # noqa: ARG002
    ) -> list[dict[str, Any]]:
        """
        Benchmarks a LLM deployment on Modal.

        :param: llm_server_type: The server to use for benchmarking, either 'vllm',
            'vllm-pd-disaggregation', 'sglang', 'tensorrt-llm', or 'tokasaurus'.
        :param: model: Name of the model to benchmark.
        :param: rate_type: The type of rate to use for benchmarking, either 'constant',
            'synchronous', or 'throughput'.
        :param: data: A configuration for emulated data (e.g.:
            'prompt_tokens=128,output_tokens=128').
        :param: gpu: The GPU to use for benchmarking.
        :param: server_region: Region to run the LLM server on.
        :param: duration: The duration of the benchmark in seconds.
        :param: llm_server_config: Configuration for the LLM server.
        :param: client_config: Configuration for the GuideLLM client.
        :param: rate: If rate_type is 'constant', specify the number of requests that
            should be made per second. If this is a list, benchmarks will be run
            sequentially at each request rate.
        :param: kwargs: Additional keyword arguments.
        """

        if llm_server_type not in LLM_SERVER_TYPES:
            msg = (
                f"Invalid value for llm_server: {llm_server_type}. Must be one of "
                f"{LLM_SERVER_TYPES}"
            )
            raise ValueError(msg)

        if llm_server_config is None:
            llm_server_config = {}

        if client_config is None:
            client_config = {}

        if not isinstance(rate_type, list):
            rate_type = [rate_type]

        if not isinstance(rate, list):
            rate = [rate]

        if len(rate_type) > 1 and len(rate) > 1:
            msg = (
                f"All benchmarks must have either the same rate type or the same rate: "
                f"{rate_type} vs. {rate}"
            )
            raise ValueError(msg)

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
        logger.info(
            (
                f"Created loader with {unique_requests} unique requests from {data}"
                if unique_requests > 0
                else f"Created loader with unknown number unique requests from {data}"
            ),
        )

        benchmark_results = []

        # Start LLM server in background
        with llm_server(
            llm_server_type,
            model=model,
            gpu=gpu,
            region=server_region,
            server_config=llm_server_config,
        ) as (llm_server_url, extra_query, startup_metrics):
            # Create backend
            backend = CustomOpenAIHTTPBackend(
                target=f"{llm_server_url}/v1",
                extra_query=extra_query,
                remove_from_body=(
                    ["max_completion_tokens", "stream"]
                    if llm_server_type == "tokasaurus"
                    else None
                ),
            )
            await backend.validate()
            logger.info("Connected to backend for model %s", backend.model)

            rates_to_run = list(itertools.product(rate_type, rate))
            logger.info("Running %d benchmarks", len(rates_to_run))

            for i, (rate_type_i, rate_i) in enumerate(rates_to_run):
                logger.info(
                    "Starting benchmark with rate type %s and rate %s",
                    rate_type_i,
                    rate_i,
                )

                profile = create_profile(rate_type=rate_type_i, rate=rate_i)
                benchmarker_kwargs = {
                    "backend": backend,
                    "request_loader": request_loader,
                    "request_loader_description": request_loader.description,
                    "benchmark_save_extras": None,
                    "processor": processor,
                    "processor_args": None,
                }

                if llm_server_type == "vllm" and client_config.get(
                    "collect_metrics",
                    False,
                ):
                    extra_query_args = urllib.parse.urlencode(extra_query)
                    metrics_url = f"{llm_server_url}/metrics?{extra_query_args}"

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
                            msg = "Current benchmark is None"
                            raise ValueError(msg)

                        benchmark_results.append(
                            {
                                "rate": rate_i,
                                "rate_type": rate_type_i,
                                "results": {
                                    **result.current_benchmark.model_dump(),
                                    **startup_metrics,
                                },
                            },
                        )

                if i != len(rates_to_run) - 1:
                    await asyncio.sleep(DELAY_BETWEEN_BENCHMARKS)

        return benchmark_results


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

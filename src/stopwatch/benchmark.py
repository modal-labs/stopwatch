import asyncio
import itertools
import logging
from collections.abc import Iterator, Mapping
from datetime import datetime, timezone
from typing import Any

import modal

from stopwatch.constants import GUIDELLM_VERSION, HOURS, SECONDS, RateType
from stopwatch.resources import app, hf_secret, results_volume, startup_metrics_dict

DELAY_BETWEEN_BENCHMARKS = 5 * SECONDS
MEMORY = 1 * 1024
NUM_CORES = 2
RESULTS_PATH = "/results"
SCALEDOWN_WINDOW = 5 * SECONDS
TIMEOUT = 4 * HOURS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

guidellm_image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .uv_pip_install(
        f"git+https://github.com/neuralmagic/guidellm.git#{GUIDELLM_VERSION}",
        "tiktoken",
    )
    .env(
        {
            "GUIDELLM__MAX_WORKER_PROCESSES": f"{NUM_CORES - 1}",
        },
    )
)

with guidellm_image.imports():
    from guidellm.backend import OpenAIHTTPBackend
    from guidellm.benchmark.benchmarker import GenerativeBenchmarker
    from guidellm.benchmark.profile import create_profile
    from guidellm.request import GenerationRequest, GenerativeRequestLoader

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


@app.cls(
    image=guidellm_image,
    secrets=[hf_secret],
    volumes={RESULTS_PATH: results_volume},
    cpu=NUM_CORES,
    memory=MEMORY,
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=TIMEOUT,
)
class GuideLLM:
    """Run benchmarks with GuideLLM."""

    @modal.method()
    async def run_benchmark(  # noqa: C901, PLR0912
        self,
        endpoint: str,
        model: str,
        rate_type: RateType | list[RateType],
        data: str,
        caller_id: str | None = None,
        duration: float | None = 120,  # 2 minutes
        client_config: Mapping[str, Any] | None = None,
        rate: float | list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Benchmarks a LLM deployment on Modal.

        :param: model: Name of the model to benchmark.
        :param: rate_type: The type of rate to use for benchmarking. If this is a list,
            benchmarks will be run sequentially at each rate type.
        :param: data: A configuration for emulated data (e.g.:
            'prompt_tokens=128,output_tokens=128').
        :param: duration: The duration of the benchmark in seconds.
        :param: client_config: Configuration for the GuideLLM client.
        :param: rate: If rate_type is RateType.constant, specify the number of requests
            that should be made per second. If this is a list, benchmarks will be run
            sequentially at each request rate.
        :param: kwargs: Additional keyword arguments.
        """

        if client_config is None:
            client_config = {}

        extra_query = client_config.get("extra_query", {})

        if caller_id is not None:
            extra_query["caller_id"] = caller_id

        # Convert rate_type to a list
        if not isinstance(rate_type, list):
            rate_type = [rate_type]

        # Convert RateTypes to strings
        for i in range(len(rate_type)):
            if isinstance(rate_type[i], RateType):
                rate_type[i] = rate_type[i].value

        # Convert rate to a list
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
        processor = client_config.get("tokenizer", model)
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

        # Create backend
        backend = OpenAIHTTPBackend(
            target=endpoint,
            extra_query=extra_query,
            remove_from_body=[
                # TensorRT-LLM returns a 400 error if max_completion_tokens is set
                "max_completion_tokens",
                *client_config.get("remove_from_body", []),
            ],
        )

        # Start the LLM server and save queuing and cold start durations
        queue_time = datetime.now(timezone.utc).timestamp()
        await backend.validate()
        connection_time = datetime.now(timezone.utc).timestamp()
        queue_duration = connection_time - queue_time

        if (
            caller_id is not None
            and (container_start_time := startup_metrics_dict.get(caller_id))
            is not None
        ):
            cold_start_duration = connection_time - container_start_time
            queue_duration -= cold_start_duration
        else:
            cold_start_duration = None

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
            benchmarker = GenerativeBenchmarker(
                backend,
                request_loader,
                request_loader.description,
                processor=processor,
            )

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
                            **result.current_benchmark.model_dump(),
                            "queue_duration": queue_duration,
                            "cold_start_duration": cold_start_duration,
                        },
                    )

            if i != len(rates_to_run) - 1:
                await asyncio.sleep(DELAY_BETWEEN_BENCHMARKS)

        return benchmark_results

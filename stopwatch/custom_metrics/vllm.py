import logging
import threading
import time
from collections.abc import AsyncGenerator
from typing import Any

import requests
from guidellm.benchmark import (
    AggregatorT,
    BenchmarkerResult,
    BenchmarkT,
    GenerativeBenchmarker,
)
from guidellm.objects import StandardBaseModel
from guidellm.scheduler import RequestT, ResponseT
from prometheus_client.metrics_core import Metric
from prometheus_client.parser import text_string_to_metric_families
from prometheus_client.samples import Sample
from pydantic import Field

from .histogram import Histogram

REFRESH_INTERVAL = 10  # 10 seconds

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class vLLMMetrics(StandardBaseModel):  # noqa: N801
    """Custom metrics reported by the vLLM server's metrics endpoint."""

    end_to_end_request_latency: Histogram = Field(
        default_factory=Histogram,
        title="vllm:e2e_request_latency_seconds",
        description="Histogram of end-to-end request latency in seconds.",
    )

    kv_cache_usage: float = Field(
        default=0,
        title="vllm:gpu_cache_usage_perc",
        description="GPU KV cache usage. 1 means 100 percent usage.",
    )

    request_decode_time: Histogram = Field(
        default_factory=Histogram,
        title="vllm:request_decode_time_seconds",
        description="Histogram of time spent in DECODE phase for requests.",
    )

    request_inference_time: Histogram = Field(
        default_factory=Histogram,
        title="vllm:request_inference_time_seconds",
        description="Histogram of time spent in RUNNING phase for requests.",
    )

    request_prefill_time: Histogram = Field(
        default_factory=Histogram,
        title="vllm:request_prefill_time_seconds",
        description="Histogram of time spent in PREFILL phase for requests.",
    )

    request_waiting_time: Histogram = Field(
        default_factory=Histogram,
        title="vllm:request_queue_time_seconds",
        description="Histogram of time spent in WAITING phase for requests.",
    )

    tokens_per_engine_step: Histogram = Field(
        default_factory=Histogram,
        title="vllm:iteration_tokens_total",
        description="Histogram of number of tokens processed per engine_step.",
    )

    time_to_first_token: Histogram = Field(
        default_factory=Histogram,
        title="vllm:time_to_first_token_seconds",
        description="Histogram of time to first token in seconds.",
    )

    time_per_output_token: Histogram = Field(
        default_factory=Histogram,
        title="vllm:time_per_output_token_seconds",
        description="Histogram of time per output token in seconds.",
    )

    def __init__(
        self,
        metric_families: list[Metric],
        first_metric_families: list[Metric],
    ) -> None:
        """
        Create a vLLM metrics object from a list of metric families. Values in the
        metric families are cumulative since the vLLM server started up, so all values
        are subtracted from the first metric families to get the actual values.
        """

        super().__init__()

        def get_histogram_samples(metric_family: Metric) -> list[Sample]:
            return list(
                filter(
                    lambda sample: sample.name.endswith("bucket"),
                    metric_family.samples,
                ),
            )

        # All histogram data is returned cumulatively since the vLLM server
        # started up, so we need to subtract all samples from the first metric
        # family to get the values for each benchmark.
        first_histogram_samples = {
            family.name: get_histogram_samples(family)
            for family in first_metric_families
            if family.type == "histogram"
        }

        vllm_keys_to_field_names = {
            field_info.title: field_name
            for field_name, field_info in self.model_fields.items()
        }

        for family in metric_families:
            if (
                family.name not in vllm_keys_to_field_names
                or family.name not in first_histogram_samples
            ):
                continue

            if family.type == "gauge":
                setattr(
                    self,
                    vllm_keys_to_field_names[family.name],
                    family.samples[0].value,
                )
            elif family.type == "histogram":
                samples = get_histogram_samples(family)

                if len(samples) != len(first_histogram_samples[family.name]):
                    msg = (
                        f"Number of samples for {family.name} changed from "
                        f"{len(first_histogram_samples[family.name])} to "
                        f"{len(samples)}"
                    )
                    raise ValueError(msg)

                counts = [
                    sample.value - first_histogram_samples[family.name][i].value
                    for i, sample in enumerate(samples)
                ]

                getattr(self, vllm_keys_to_field_names[family.name]).set_data(
                    # The counts are cumulative, so we need to subtract the
                    # counts from the previous bucket to get the actual count
                    # for each bucket.
                    [x - counts[i - 1] if i > 0 else x for i, x in enumerate(counts)],
                    [
                        0,
                        *[
                            (
                                float(sample.labels["le"])
                                if sample.labels["le"] != "+Inf"
                                else 999999
                            )
                            for sample in samples
                        ],
                    ],
                )
            else:
                msg = f"Unsupported metric type: {family.type}"
                raise ValueError(msg)


class GenerativeBenchmarkerWithvLLMMetrics(GenerativeBenchmarker):
    """A benchmarker that periodically fetches metrics from the vLLM server."""

    def __init__(self, vllm_metrics_url: str, **kwargs: dict[str, Any]) -> None:
        """
        Create a benchmarker that periodically fetches metrics from the vLLM server.

        :param: vllm_metrics_url: The URL of the vLLM server metrics endpoint.
        :param: kwargs: Additional keyword arguments to pass to the benchmarker.
        """

        super().__init__(**kwargs)
        self.vllm_metrics_url = vllm_metrics_url

    async def run(
        self,
        **kwargs: dict[str, Any],
    ) -> AsyncGenerator[
        BenchmarkerResult[AggregatorT, BenchmarkT, RequestT, ResponseT],
        None,
    ]:
        """
        Run the executor and periodically fetch metrics from the vLLM server while
        benchmarks are running.
        """

        def reset() -> tuple[list, threading.Event, threading.Thread]:
            vllm_metrics = []
            stop_signal = threading.Event()
            stats_thread = threading.Thread(
                target=self.refresh_vllm_stats,
                args=(
                    self.vllm_metrics_url,
                    vllm_metrics,
                    stop_signal,
                ),
            )
            stats_thread.start()
            return vllm_metrics, stop_signal, stats_thread

        # Start fetching metrics before the first benchmark starts
        vllm_metric_families, stop_signal, stats_thread = reset()

        async for result in super().run(**kwargs):
            if result.type_ == "benchmark_compiled":
                # If the last benchmark has completed, stop fetching metrics
                stop_signal.set()
                stats_thread.join()

                # Use the first metric family that has every metric
                metrics_per_metric_family = [
                    len(metric_family) for metric_family in vllm_metric_families
                ]
                i = 0
                while i < len(metrics_per_metric_family):
                    if metrics_per_metric_family[i] == metrics_per_metric_family[-1]:
                        break
                    i += 1

                if "vllm_metrics" not in result.current_benchmark.extras:
                    result.current_benchmark.extras["vllm_metrics"] = []

                result.current_benchmark.extras["vllm_metrics"].extend(
                    [
                        vLLMMetrics(
                            metrics,
                            first_metric_families=vllm_metric_families[i],
                        )
                        for metrics in vllm_metric_families
                    ],
                )

            yield result

            if result.type_ == "benchmark_compiled":
                # If a new benchmark is about to start, start fetching
                # metrics again
                vllm_metric_families, stop_signal, stats_thread = reset()

        stop_signal.set()
        stats_thread.join()

    def refresh_vllm_stats(
        self,
        metrics_url: str,
        metrics: list,
        stop_signal: threading.Event,
    ) -> None:
        """
        Periodically fetches metrics from the vLLM server while benchmarks are
        running.

        :param: metrics_url: The URL of the vLLM server metrics endpoint.
        :param: metrics: A list to store the fetched metrics.
        :param: stop_signal: A signal to stop the metrics refresh.
        """

        while not stop_signal.is_set():
            res = requests.get(metrics_url)

            try:
                metrics.append(list(text_string_to_metric_families(res.text)))
            except Exception:
                logger.exception("Error parsing metrics")

            time.sleep(REFRESH_INTERVAL)

from typing import AsyncGenerator, ClassVar, List, Optional
import threading
import time
import urllib.request

from guidellm.core.report import GuidanceReport
from guidellm.core.result import TextGenerationBenchmark, TextGenerationBenchmarkReport
from guidellm.core.serializable import Serializable
from guidellm.scheduler import Scheduler, SchedulerResult
from prometheus_client.parser import text_string_to_metric_families
from pydantic import Field

from .histogram import Histogram


REFRESH_INTERVAL = 10  # 10 seconds


class SchedulerWithvLLMMetrics(Scheduler):

    _vllm_metrics_url: ClassVar[Optional[str]] = None

    async def run(self) -> AsyncGenerator[SchedulerResult, None]:
        """
        Runs the executor and periodically fetches metrics from the vLLM server
        while benchmarks are running.
        """

        def reset():
            vllm_metrics = []
            stop_signal = threading.Event()
            stats_thread = threading.Thread(
                target=self.refresh_vllm_stats,
                args=(
                    self._vllm_metrics_url,
                    vllm_metrics,
                    stop_signal,
                ),
            )
            stats_thread.start()
            return vllm_metrics, stop_signal, stats_thread

        # Start fetching metrics before the first benchmark starts
        vllm_metric_families, stop_signal, stats_thread = reset()

        async for result in super().run():
            if result.completed:
                # If the last benchmark has completed, stop fetching metrics
                stop_signal.set()
                stats_thread.join()
                result.benchmark.vllm_metrics.extend(
                    [
                        vLLMMetrics(
                            metrics, first_metric_families=vllm_metric_families[0]
                        )
                        for metrics in vllm_metric_families
                    ]
                )

            yield result

            if result.completed:
                # If a new benchmark is about to start, start fetching
                # metrics again
                vllm_metric_families, stop_signal, stats_thread = reset()

        stop_signal.set()
        stats_thread.join()

    def refresh_vllm_stats(
        self, metrics_url: str, metrics: list, stop_signal: threading.Event
    ):
        """
        Periodically fetches metrics from the vLLM server while benchmarks are
        running.

        Args:
            metrics_url (str): The URL of the vLLM server metrics endpoint.
            metrics (list): A list to store the fetched metrics.
            stop_signal (threading.Event): A signal to stop the metrics refresh.
        """

        while not stop_signal.is_set():
            with urllib.request.urlopen(metrics_url) as response:
                r_metrics = text_string_to_metric_families(response.read().decode())

            metrics.append(list(r_metrics))
            time.sleep(REFRESH_INTERVAL)


class vLLMMetrics(Serializable):

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

    def __init__(self, metric_families, first_metric_families):
        super().__init__()

        def get_histogram_samples(metric_family):
            return list(
                filter(
                    lambda sample: sample.name.endswith("bucket"), metric_family.samples
                )
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
            if family.name not in vllm_keys_to_field_names:
                continue

            if family.type == "gauge":
                setattr(
                    self, vllm_keys_to_field_names[family.name], family.samples[0].value
                )
            elif family.type == "histogram":
                samples = get_histogram_samples(family)

                assert len(samples) == len(first_histogram_samples[family.name])

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
                raise ValueError(f"Unsupported metric type: {family.type}")


class TextGenerationBenchmarkWithvLLMMetrics(TextGenerationBenchmark):

    vllm_metrics: List[vLLMMetrics] = Field(
        default_factory=list,
        description="The vLLM metric families sampled while taking the benchmark.",
    )


class TextGenerationBenchmarkReportWithvLLMMetrics(TextGenerationBenchmarkReport):

    benchmarks: List[TextGenerationBenchmarkWithvLLMMetrics] = Field(
        default_factory=list,
        description="The benchmarks of text generation requests.",
    )


class GuidanceReportWithvLLMMetrics(GuidanceReport):

    benchmarks: List[TextGenerationBenchmarkReportWithvLLMMetrics] = Field(
        default_factory=list,
        description="The list of benchmark reports.",
    )


def vllm_monkey_patch(metrics_url: str):
    import guidellm.executor.base
    import guidellm.scheduler.base

    guidellm.executor.base.Scheduler = SchedulerWithvLLMMetrics
    guidellm.main.GuidanceReport = GuidanceReportWithvLLMMetrics
    guidellm.scheduler.base.TextGenerationBenchmark = (
        TextGenerationBenchmarkWithvLLMMetrics
    )

    SchedulerWithvLLMMetrics._vllm_metrics_url = metrics_url

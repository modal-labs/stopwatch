from typing import AsyncGenerator, ClassVar, List, Optional
import requests
import threading
import time

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
            r = requests.get(metrics_url)
            r_metrics = text_string_to_metric_families(r.text)
            metrics.append(list(r_metrics))
            time.sleep(REFRESH_INTERVAL)


class vLLMMetrics(Serializable):

    kv_cache_usage: float = Field(
        default=0,
        description="GPU KV cache usage. 1 means 100 percent usage.",
    )

    time_per_output_token: Histogram = Field(
        default_factory=Histogram,
        description="Histogram of time per output token in seconds.",
    )

    def __init__(self, metric_families, first_metric_families):
        super().__init__()

        # TPOT is returned cumulatively since the vLLM server starts up, so we
        # need to subtract all samples from the first metric family to get the
        # values for this benchmark.
        first_tpot_family_samples = list(
            filter(
                lambda sample: sample.name
                == "vllm:time_per_output_token_seconds_bucket",
                next(
                    filter(
                        lambda family: family.name
                        == "vllm:time_per_output_token_seconds",
                        first_metric_families,
                    )
                ).samples,
            )
        )

        for family in metric_families:
            if family.name == "vllm:gpu_cache_usage_perc":
                self.kv_cache_usage = family.samples[0].value
            elif family.name == "vllm:time_per_output_token_seconds":
                samples = list(
                    filter(
                        lambda sample: sample.name
                        == "vllm:time_per_output_token_seconds_bucket",
                        family.samples,
                    )
                )

                assert len(samples) == len(first_tpot_family_samples)

                counts = [
                    sample.value - first_tpot_family_samples[i].value
                    for i, sample in enumerate(samples)
                ]

                self.time_per_output_token.set_data(
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

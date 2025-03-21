from typing import List
import itertools

import numpy as np
from sqlalchemy import Column, DateTime, Float, Integer, JSON, String
from sqlalchemy.sql import func

from .base import Base


def histogram_median(bins, counts):
    if len(bins) == len(counts) == 0:
        return None

    assert len(bins) == len(counts) + 1, f"len({bins}) != len({counts}) + 1"

    total = sum(counts)
    half = total / 2

    # Walk through the histogram until we reach or exceed the halfway point
    cumulative = 0
    for i, count in enumerate(counts):
        new_cumulative = cumulative + count

        if new_cumulative >= total / 2:
            # Linearly interpolate within the bin
            fraction = (half - cumulative) / count if count > 0 else 0
            bin_width = bins[i + 1] - bins[i]
            return bins[i] + fraction * bin_width

        cumulative = new_cumulative


class BenchmarkDefaults:
    GPU = "H100"
    REGION = "us-ashburn-1"
    LLM_SERVER_CONFIGS = {
        "vllm": {
            "docker_tag": "v0.7.3",
            "env_vars": {},
            "extra_args": [],
        },
        "trtllm": {},
    }


class Benchmark(Base):
    __tablename__ = "benchmarks"

    # Metadata
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=func.now())
    function_call_id = Column(String)
    repeat_index = Column(Integer, nullable=False, default=0)
    group_id = Column(String, nullable=False)

    # Parameters
    llm_server_type = Column(String, nullable=False)
    llm_server_config = Column(JSON, nullable=False)
    rate = Column(Float)
    rate_type = Column(String, nullable=False)
    model = Column(String, nullable=False)
    data = Column(String, nullable=False)
    gpu = Column(String, default=BenchmarkDefaults.GPU, nullable=False)
    region = Column(String, default=BenchmarkDefaults.REGION, nullable=False)

    # Results
    start_time = Column(Float)
    end_time = Column(Float)
    duration = Column(Float)
    completed_request_count = Column(Integer)
    completed_request_rate = Column(Float)
    generated_tokens = Column(Integer)
    tpot_median = Column(Float)

    kv_cache_usage_mean = Column(Float)
    prompt_tokens = Column(Integer)

    itl_mean = Column(Float)
    itl_p50 = Column(Float)
    itl_p90 = Column(Float)
    itl_p95 = Column(Float)
    itl_p99 = Column(Float)
    ttft_mean = Column(Float)
    ttft_p50 = Column(Float)
    ttft_p90 = Column(Float)
    ttft_p95 = Column(Float)
    ttft_p99 = Column(Float)
    ttlt_mean = Column(Float)
    ttlt_p50 = Column(Float)
    ttlt_p90 = Column(Float)
    ttlt_p95 = Column(Float)
    ttlt_p99 = Column(Float)

    def get_config(self):
        return {
            "rate": self.rate,
            "rate_type": self.rate_type,
            "model": self.model,
            "data": self.data,
            "gpu": self.gpu,
            "region": self.region,
            "llm_server_type": self.llm_server_type,
            "llm_server_config": self.llm_server_config,
        }

    def save_results(self, results, vllm_metrics=None):
        if not results:
            raise ValueError("No results to save")

        self.start_time = results[0]["start_time"]
        self.end_time = results[-1]["end_time"]
        self.duration = self.end_time - self.start_time
        self.completed_request_count = len(results)
        self.completed_request_rate = self.completed_request_count / self.duration

        data_config = {
            k: int(v) for param in self.data.split(",") for k, v in [param.split("=")]
        }

        self.prompt_tokens = data_config.get("prompt_tokens", 0)
        self.generated_tokens = data_config.get("generated_tokens", 0)

        ttft_distribution = [
            result["first_token_time"]
            for result in results
            if result["first_token_time"] is not None
        ]
        ttlt_distribution = [
            result["end_time"] - result["start_time"] for result in results
        ]
        itl_distribution = [
            decode_time
            for result in results
            for decode_time in result["decode_times"]["data"]
        ]

        for key, distribution, statistic in itertools.product(
            ["itl", "ttft", "ttlt"],
            [itl_distribution, ttft_distribution, ttlt_distribution],
            ["mean", 50, 90, 95, 99],
        ):
            if statistic == "mean":
                setattr(self, f"{key}_mean", np.mean(distribution))
            else:
                setattr(
                    self, f"{key}_p{statistic}", np.percentile(distribution, statistic)
                )

        # Save vLLM metrics
        if vllm_metrics is not None:
            self.kv_cache_usage_mean = 100 * np.mean(
                [metrics["kv_cache_usage"] for metrics in vllm_metrics]
            )
            self.tpot_median = histogram_median(
                vllm_metrics[-1]["time_per_output_token"]["bins"],
                vllm_metrics[-1]["time_per_output_token"]["data"],
            )


# TODO: Remove this class, use inheritance
class AveragedBenchmark(Base):
    __tablename__ = "averaged_benchmarks"

    # Metadata
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=func.now())
    group_id = Column(String, nullable=False)

    # Parameters
    llm_server_type = Column(String, nullable=False)
    llm_server_config = Column(JSON, nullable=False)
    rate = Column(Float)
    rate_type = Column(String, nullable=False)
    model = Column(String, nullable=False)
    data = Column(String, nullable=False)
    gpu = Column(String, default=BenchmarkDefaults.GPU, nullable=False)
    region = Column(String, default=BenchmarkDefaults.REGION, nullable=False)

    # Results
    start_time = Column(Float)
    end_time = Column(Float)
    duration = Column(Float)
    completed_request_count = Column(Integer)
    completed_request_rate = Column(Float)
    generated_tokens = Column(Integer)
    tpot_median = Column(Float)

    kv_cache_usage_mean = Column(Float)
    prompt_tokens = Column(Integer)

    itl_mean = Column(Float)
    itl_p50 = Column(Float)
    itl_p90 = Column(Float)
    itl_p95 = Column(Float)
    itl_p99 = Column(Float)
    ttft_mean = Column(Float)
    ttft_p50 = Column(Float)
    ttft_p90 = Column(Float)
    ttft_p95 = Column(Float)
    ttft_p99 = Column(Float)
    ttlt_mean = Column(Float)
    ttlt_p50 = Column(Float)
    ttlt_p90 = Column(Float)
    ttlt_p95 = Column(Float)
    ttlt_p99 = Column(Float)

    def __init__(self, benchmarks: List[Benchmark], group_id: str):
        super().__init__()

        for key in [
            "rate",
            "rate_type",
            "model",
            "data",
            "gpu",
            "region",
            "llm_server_type",
            "llm_server_config",
        ]:
            setattr(self, key, benchmarks[0].__getattribute__(key))

        self.group_id = group_id

        for key in [
            "duration",
            "completed_request_count",
            "completed_request_rate",
            "tpot_median",
            "kv_cache_usage_mean",
            "itl_mean",
            "itl_p50",
            "itl_p90",
            "itl_p95",
            "itl_p99",
            "ttft_mean",
            "ttft_p50",
            "ttft_p90",
            "ttft_p95",
            "ttft_p99",
            "ttlt_mean",
            "ttlt_p50",
            "ttlt_p90",
            "ttlt_p95",
            "ttlt_p99",
        ]:
            # TODO: Think about what to do with None values (in tpot_median)
            data = [getattr(b, key) for b in benchmarks if getattr(b, key) is not None]

            if len(data) == 0:
                continue

            setattr(
                self,
                key,
                np.median(data),
            )

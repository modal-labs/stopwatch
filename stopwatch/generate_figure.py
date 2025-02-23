import modal

from .benchmark import get_benchmark_fingerprint
from .resources import app, figures_volume, results_dict, results_volume
from .run_benchmark import run_benchmark

FIGURES_PATH = "/figures"
RESULTS_PATH = "/results"
TIMEOUT = 60 * 60  # 1 hour

figures_image = modal.Image.debian_slim().pip_install("matplotlib", "numpy", "pandas")

with figures_image.imports():
    from typing import Any, Dict, List
    import json
    import os

    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
    import pandas as pd


def histogram_median(bins, counts):
    assert len(bins) == len(counts) + 1
    assert sum(counts) > 0

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


@app.function(
    image=figures_image,
    volumes={RESULTS_PATH: results_volume, FIGURES_PATH: figures_volume},
    timeout=TIMEOUT,
)
def generate_figure(benchmarks: List[Dict[str, Any]], title: str):
    benchmarks = benchmarks.copy()

    # Create fingerprints for each benchmark. This allows us to check if the
    # benchmark has already been run recently, in which case we don't need
    # to run it again.
    for i, benchmark in enumerate(benchmarks):
        benchmarks[i]["fingerprint"] = get_benchmark_fingerprint(**benchmark["config"])

    # Run benchmarks that aren't already cached
    pending_benchmarks = []

    for i, benchmark in enumerate(benchmarks):
        if benchmark["fingerprint"] in results_dict and not benchmark.get(
            "recompute", False
        ):
            continue

        fc = run_benchmark.spawn(benchmark["vllm_deployment_id"], **benchmark["config"])
        pending_benchmarks.append(fc)

    # Wait for all newly run benchmarks to finish
    for fc in pending_benchmarks:
        fc.get()

    results_volume.reload()

    # Process results
    for i, benchmark in enumerate(benchmarks):
        results_path = os.path.join(
            RESULTS_PATH, f"{results_dict[benchmark['fingerprint']]}.json"
        )
        results_data = json.load(open(results_path))
        df = pd.DataFrame(results_data["benchmarks"][0]["benchmarks"])
        df = df[df["mode"] != "throughput"]

        df["duration"] = df.apply(
            lambda x: x["results"][-1]["end_time"] - x["results"][0]["start_time"],
            axis=1,
        )
        df["requests_per_second"] = df.apply(
            lambda x: len(x["results"]) / x["duration"], axis=1
        )

        # Save inter-token latency distribution
        df["itl_distribution"] = df.apply(
            lambda x: [
                decode * 1000
                for result in x["results"]
                for decode in result["decode_times"]["data"]
            ],
            axis=1,
        )
        df["itl_mean"] = df.apply(lambda x: np.mean(x["itl_distribution"]), axis=1)

        # Save time-to-first-token distribution
        df["ttft_distribution"] = df.apply(
            lambda x: [
                y["first_token_time"] * 1000
                for y in x["results"]
                if y["first_token_time"] is not None
            ],
            axis=1,
        )
        df["ttft_mean"] = df.apply(lambda x: np.mean(x["ttft_distribution"]), axis=1)
        for j in range(5, 96):
            df[f"ttft_p{j}"] = df.apply(
                lambda x: np.percentile(x["ttft_distribution"], j), axis=1
            )

        # Save tpot median from vLLM metrics
        df["tpot_median"] = df.apply(
            lambda x: 1000
            * histogram_median(
                x["vllm_metrics"][-1]["time_per_output_token"]["bins"],
                x["vllm_metrics"][-1]["time_per_output_token"]["data"],
            ),
            axis=1,
        )

        # Save KV cache usage from vLLM metrics
        df["kv_cache_usage_mean"] = df.apply(
            lambda x: np.mean([y["kv_cache_usage"] for y in x["vllm_metrics"]]),
            axis=1,
        )

        df = df.sort_values("requests_per_second", ascending=True)
        benchmarks[i]["df"] = df

    # Create figure
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(5, 6))

    for benchmark in benchmarks:
        axs[0].plot(
            benchmark["df"]["requests_per_second"],
            benchmark["df"]["tpot_median"],
            label=benchmark["name"],
        )

        (ttft_line,) = axs[1].plot(
            benchmark["df"]["requests_per_second"],
            benchmark["df"]["ttft_mean"],
            label=benchmark["name"],
        )
        for i in range(5, 96):
            axs[1].plot(
                benchmark["df"]["requests_per_second"],
                benchmark["df"][f"ttft_p{i}"],
                color=ttft_line.get_color(),
                alpha=0.1,
            )

        axs[2].plot(
            benchmark["df"]["requests_per_second"],
            benchmark["df"]["kv_cache_usage_mean"],
            label=benchmark["name"],
        )

    # Configure TPOT plot
    axs[0].set_ylabel("Time per output token (ms)")

    # Configure TTFT plot
    axs[1].set_ylabel("Time to first token (ms)")
    axs[1].set_yscale("log")
    axs[1].yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, p: f"{int(x)}" if x >= 1 else f"{x:.1f}")
    )

    # Configure KV cache usage plot
    axs[2].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x*100:.1f}%"))
    axs[2].set_ylabel("KV cache usage")

    # Configure legend and outside labels
    axs[0].set_title(title)
    axs[-1].set_xlabel("Requests per second")

    for ax in axs:
        ax.grid(alpha=0.5, linestyle="--")

    handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    fig.savefig(
        os.path.join(FIGURES_PATH, f"{modal.current_function_call_id()}.png"),
        dpi=300,
        bbox_inches="tight",
    )

    figures_volume.commit()
    return f"{modal.current_function_call_id()}.png"

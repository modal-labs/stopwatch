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
    import numpy as np
    import pandas as pd


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
        if benchmark["fingerprint"] in results_dict:
            continue

        fc = run_benchmark.spawn(**benchmark["config"])
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

        # Save time-to-first-token distribution
        df["ttft_distribution"] = df.apply(
            lambda x: [
                y["first_token_time"] * 1000
                for y in x["results"]
                if y["first_token_time"] is not None
            ],
            axis=1,
        )

        # Calculate metrics
        for metric in ["itl", "ttft"]:
            df[f"{metric}_mean"] = df.apply(
                lambda x: np.mean(x[f"{metric}_distribution"]), axis=1
            )
            df[f"{metric}_p5"] = df.apply(
                lambda x: np.percentile(x[f"{metric}_distribution"], 5), axis=1
            )
            df[f"{metric}_p95"] = df.apply(
                lambda x: np.percentile(x[f"{metric}_distribution"], 95), axis=1
            )

        df = df.sort_values("requests_per_second", ascending=True)
        benchmarks[i]["df"] = df

    # Create figure
    fig, axs = plt.subplots(2, 1, sharex=True)

    for benchmark in benchmarks:
        (line,) = axs[0].plot(
            benchmark["df"]["requests_per_second"],
            benchmark["df"]["itl_mean"],
            label=benchmark["name"],
        )
        axs[0].scatter(
            benchmark["df"]["requests_per_second"],
            benchmark["df"]["itl_mean"],
            color=line.get_color(),
            label="_nolegend_",
            s=5,
            marker="|",
        )
        axs[0].fill_between(
            benchmark["df"]["requests_per_second"],
            benchmark["df"]["itl_p5"],
            benchmark["df"]["itl_p95"],
            alpha=0.2,
        )

        (line,) = axs[1].plot(
            benchmark["df"]["requests_per_second"],
            benchmark["df"]["ttft_mean"],
            label=benchmark["name"],
        )
        axs[1].scatter(
            benchmark["df"]["requests_per_second"],
            benchmark["df"]["ttft_mean"],
            color=line.get_color(),
            label="_nolegend_",
            s=5,
            marker="|",
        )
        axs[1].fill_between(
            benchmark["df"]["requests_per_second"],
            benchmark["df"]["ttft_p5"],
            benchmark["df"]["ttft_p95"],
            alpha=0.2,
        )

    axs[0].set_title(title)
    axs[-1].set_xlabel("Requests per second")
    axs[0].set_ylabel("Time per output token (ms)")
    axs[1].set_ylabel("Time to first token (ms)")

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

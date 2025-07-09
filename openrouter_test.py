#!/usr/bin/env python3
"""
Simple GuideLLM script for testing models on OpenRouter.

Usage:
    python openrouter_test.py --model anthropic/claude-3-haiku --data "prompt_tokens=512,output_tokens=128"
"""

import argparse
import asyncio
import os
import statistics
import sys

from guidellm.benchmark.benchmarker import GenerativeBenchmarker
from guidellm.benchmark.profile import create_profile
from typing import Any
from transformers import AutoTokenizer

# Import the existing custom classes from the project
from stopwatch.run_benchmark import (
    CustomGenerativeRequestLoader,
    CustomOpenAIHTTPBackend,
)


def parse_data_config(data_str: str) -> dict[str, int]:
    """Parse data configuration string into dictionary."""
    data_config = {}
    for param in data_str.split(","):
        key, value = param.split("=")
        data_config[key.strip()] = int(value.strip())
    return data_config


def build_provider_config(provider_name: str | None) -> dict[str, Any] | None:
    """Build provider configuration for OpenRouter API.
    
    Args:
        provider_name: Name of the provider to use exclusively, or None for default behavior
        
    Returns:
        Provider configuration dict or None for default behavior
    """
    if provider_name is None:
        return None
    
    # Use only the specified provider with no fallbacks
    return {"only": [provider_name]}


def calculate_ttlt(requests: list[dict[str, Any]]) -> list[float]:
    """Calculate Time to Last Token (TTLT) from individual requests."""
    ttlt_values = []
    for req in requests:
        if "first_token_time" in req and "last_token_time" in req:
            # TTLT = time from start to last token
            ttlt = (req["last_token_time"] - req["start_time"]) * 1000  # Convert to ms
            ttlt_values.append(ttlt)
    return ttlt_values


def extract_metrics_from_result(result: dict[str, Any]) -> dict[str, Any]:
    """Extract TTFT, TTLT, and ITL metrics from a single benchmark result."""
    metrics = result.get("results", {}).get("metrics", {})

    # Extract successful requests for individual-level metrics
    successful_requests = []
    if "requests" in result.get("results", {}):
        successful_requests = result["results"]["requests"].get("successful", [])

    # Time to First Token (TTFT)
    ttft_data = metrics.get("time_to_first_token_ms", {}).get("successful", {})

    # Inter-Token Latency (ITL)
    itl_data = metrics.get("inter_token_latency_ms", {}).get("successful", {})

    # Time to Last Token (TTLT) - calculate from individual requests
    ttlt_values = calculate_ttlt(successful_requests)

    return {
        "ttft": ttft_data,
        "itl": itl_data,
        "ttlt_values": ttlt_values,
        "rate_type": result.get("rate_type", "unknown"),
        "rate": result.get("rate", "unknown"),
        "successful_requests": len(successful_requests),
    }


def format_percentile_table(all_metrics: list[dict[str, Any]]) -> str:
    """Format metrics into a summary table."""
    table = []

    # Header
    header = [
        "Metric",
        "Mean",
        "P50",
        "P90",
        "P95",
        "P99",
        "Min",
        "Max",
        "Std Dev",
        "Count",
    ]
    table.append(header)
    table.append(["-" * len(h) for h in header])

    # Process each benchmark result
    for i, metrics in enumerate(all_metrics):
        rate_info = f"Rate: {metrics['rate']} ({metrics['rate_type']})"
        table.append(
            [
                f"--- Benchmark {i + 1} ({rate_info}) ---",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ]
        )

        # TTFT metrics
        ttft = metrics["ttft"]
        if ttft:
            table.append(
                [
                    "TTFT (ms)",
                    f"{ttft.get('mean', 0):.2f}",
                    f"{ttft.get('percentiles', {}).get('p50', 0):.2f}",
                    f"{ttft.get('percentiles', {}).get('p90', 0):.2f}",
                    f"{ttft.get('percentiles', {}).get('p95', 0):.2f}",
                    f"{ttft.get('percentiles', {}).get('p99', 0):.2f}",
                    f"{ttft.get('min', 0):.2f}",
                    f"{ttft.get('max', 0):.2f}",
                    f"{ttft.get('std_dev', 0):.2f}",
                    f"{ttft.get('count', 0)}",
                ]
            )

        # ITL metrics
        itl = metrics["itl"]
        if itl:
            table.append(
                [
                    "ITL (ms)",
                    f"{itl.get('mean', 0):.2f}",
                    f"{itl.get('percentiles', {}).get('p50', 0):.2f}",
                    f"{itl.get('percentiles', {}).get('p90', 0):.2f}",
                    f"{itl.get('percentiles', {}).get('p95', 0):.2f}",
                    f"{itl.get('percentiles', {}).get('p99', 0):.2f}",
                    f"{itl.get('min', 0):.2f}",
                    f"{itl.get('max', 0):.2f}",
                    f"{itl.get('std_dev', 0):.2f}",
                    f"{itl.get('count', 0)}",
                ]
            )

        # TTLT metrics (calculated from individual requests)
        ttlt_values = metrics["ttlt_values"]
        if ttlt_values:
            ttlt_mean = statistics.mean(ttlt_values)
            ttlt_median = statistics.median(ttlt_values)
            ttlt_std = statistics.stdev(ttlt_values) if len(ttlt_values) > 1 else 0
            ttlt_min = min(ttlt_values)
            ttlt_max = max(ttlt_values)

            # Calculate percentiles
            sorted_ttlt = sorted(ttlt_values)
            n = len(sorted_ttlt)
            p90_idx = int(0.90 * n)
            p95_idx = int(0.95 * n)
            p99_idx = int(0.99 * n)

            table.append(
                [
                    "TTLT (ms)",
                    f"{ttlt_mean:.2f}",
                    f"{ttlt_median:.2f}",
                    f"{sorted_ttlt[p90_idx]:.2f}",
                    f"{sorted_ttlt[p95_idx]:.2f}",
                    f"{sorted_ttlt[p99_idx]:.2f}",
                    f"{ttlt_min:.2f}",
                    f"{ttlt_max:.2f}",
                    f"{ttlt_std:.2f}",
                    f"{n}",
                ]
            )

        table.append(["", "", "", "", "", "", "", "", "", ""])  # Empty row separator

    # Format as aligned table
    col_widths = [max(len(str(row[i])) for row in table) for i in range(len(table[0]))]
    formatted_table = []

    for row in table:
        formatted_row = " | ".join(
            f"{str(cell):<{col_widths[i]}}" for i, cell in enumerate(row)
        )
        formatted_table.append(formatted_row)

    return "\n".join(formatted_table)


def calculate_aggregate_stats(all_metrics: list[dict[str, Any]]) -> str:
    """Calculate aggregate statistics across all benchmarks."""
    all_ttft = []
    all_itl = []
    all_ttlt = []

    for metrics in all_metrics:
        # Collect TTFT values
        ttft_data = metrics["ttft"]
        if ttft_data and "percentiles" in ttft_data:
            all_ttft.append(
                {
                    "mean": ttft_data.get("mean", 0),
                    "p50": ttft_data.get("percentiles", {}).get("p50", 0),
                    "p90": ttft_data.get("percentiles", {}).get("p90", 0),
                    "p95": ttft_data.get("percentiles", {}).get("p95", 0),
                    "p99": ttft_data.get("percentiles", {}).get("p99", 0),
                }
            )

        # Collect ITL values
        itl_data = metrics["itl"]
        if itl_data and "percentiles" in itl_data:
            all_itl.append(
                {
                    "mean": itl_data.get("mean", 0),
                    "p50": itl_data.get("percentiles", {}).get("p50", 0),
                    "p90": itl_data.get("percentiles", {}).get("p90", 0),
                    "p95": itl_data.get("percentiles", {}).get("p95", 0),
                    "p99": itl_data.get("percentiles", {}).get("p99", 0),
                }
            )

        # Collect TTLT values
        ttlt_values = metrics["ttlt_values"]
        if ttlt_values:
            sorted_ttlt = sorted(ttlt_values)
            n = len(sorted_ttlt)
            all_ttlt.append(
                {
                    "mean": statistics.mean(ttlt_values),
                    "p50": statistics.median(ttlt_values),
                    "p90": sorted_ttlt[int(0.90 * n)],
                    "p95": sorted_ttlt[int(0.95 * n)],
                    "p99": sorted_ttlt[int(0.99 * n)],
                }
            )

    if not all_ttft:
        return "No data available for aggregate statistics"

    # Calculate averages across benchmarks
    result = "=== AGGREGATE STATISTICS ACROSS ALL BENCHMARKS ===\n\n"

    for metric_name, metric_data in [
        ("TTFT", all_ttft),
        ("ITL", all_itl),
        ("TTLT", all_ttlt),
    ]:
        if metric_data:
            result += f"{metric_name} (ms) - Average across {len(metric_data)} benchmark(s):\n"
            result += (
                f"  Mean: {statistics.mean([d['mean'] for d in metric_data]):.2f}\n"
            )
            result += (
                f"  P50:  {statistics.mean([d['p50'] for d in metric_data]):.2f}\n"
            )
            result += (
                f"  P90:  {statistics.mean([d['p90'] for d in metric_data]):.2f}\n"
            )
            result += (
                f"  P95:  {statistics.mean([d['p95'] for d in metric_data]):.2f}\n"
            )
            result += (
                f"  P99:  {statistics.mean([d['p99'] for d in metric_data]):.2f}\n"
            )
            result += "\n"

    return result


async def main():
    """Main function to run the OpenRouter benchmark."""
    parser = argparse.ArgumentParser(
        description="Test models on OpenRouter using GuideLLM"
    )
    parser.add_argument(
        "--url",
        default="https://openrouter.ai/api",
        help="OpenRouter API URL (default: https://openrouter.ai/api)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name on OpenRouter (e.g., anthropic/claude-3-haiku)",
    )
    parser.add_argument(
        "--data",
        default="prompt_tokens=512,output_tokens=128",
        help="Data configuration (default: prompt_tokens=512,output_tokens=128)",
    )
    parser.add_argument(
        "--api-key",
        help="OpenRouter API key (or set OPENROUTER_API_KEY environment variable)",
    )
    parser.add_argument(
        "--tokenizer",
        help="HuggingFace tokenizer model name (defaults to using the model name)",
    )
    parser.add_argument(
        "--rate-type",
        choices=["synchronous", "constant"],
        default="synchronous",
        help="Rate type for load testing (default: synchronous)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        help="Rate for constant rate type (requests per second)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Duration of the benchmark in seconds (default: 60)",
    )
    parser.add_argument(
        "--provider",
        help="Specific provider to use (e.g., 'anthropic', 'openai'). If not specified, uses OpenRouter's default routing.",
    )

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print(
            "Error: OpenRouter API key required. Set --api-key or OPENROUTER_API_KEY environment variable."
        )
        sys.exit(1)

    # Parse data configuration
    data_config = parse_data_config(args.data)
    print(f"Data configuration: {data_config}")
    
    # Build provider configuration
    provider_config = build_provider_config(args.provider)
    if provider_config:
        print(f"Provider: {args.provider} (exclusive)")
    else:
        print("Provider: default routing")

    # Initialize tokenizer
    tokenizer_model = args.tokenizer or args.model
    print(f"Loading tokenizer: {tokenizer_model}")
    try:
        processor = AutoTokenizer.from_pretrained(tokenizer_model)
    except Exception as e:
        print(f"Error loading tokenizer {tokenizer_model}: {e}")
        print("Falling back to a generic tokenizer...")
        processor = AutoTokenizer.from_pretrained("gpt2")

    # Create request loader using the existing CustomGenerativeRequestLoader
    print("Creating request loader...")
    extra_body = {}
    if provider_config:
        extra_body["provider"] = provider_config
    
    request_loader = CustomGenerativeRequestLoader(
        data=args.data,
        data_args=None,
        processor=processor,
        processor_args=None,
        shuffle=False,
        iter_type="infinite",
        random_seed=42,
        extra_body=extra_body,
        use_chat_completions=True,  # OpenRouter typically uses chat completions
    )

    unique_requests = request_loader.num_unique_items(raise_err=False)
    print(f"Created loader with {unique_requests} unique requests")

    # Setup backend using the existing CustomOpenAIHTTPBackend
    print("Setting up OpenRouter backend...")
    backend = CustomOpenAIHTTPBackend(
        target=f"{args.url}/v1",
        model=args.model,
        api_key=api_key,
    )

    # Validate backend
    print("Validating backend connection...")
    try:
        await backend.validate()
        print(f"Connected to backend for model {backend.model}")
    except Exception as e:
        print(f"Error validating backend: {e}")
        sys.exit(1)

    # Create profile
    print("Creating benchmark profile...")
    if args.rate_type == "constant" and args.rate is None:
        print("Error: --rate is required when --rate-type is constant")
        sys.exit(1)

    profile = create_profile(
        rate_type=args.rate_type,
        rate=args.rate if args.rate_type == "constant" else None,
    )

    # Run benchmark
    print(f"Running benchmark for {args.duration} seconds...")
    benchmarker = GenerativeBenchmarker(
        backend=backend,
        request_loader=request_loader,
        request_loader_description=request_loader.description,
        benchmark_save_extras=None,
        processor=processor,
        processor_args=None,
    )

    results = []
    try:
        async for result in benchmarker.run(
            profile=profile,
            max_number_per_strategy=None,
            max_duration_per_strategy=args.duration,
            warmup_percent_per_strategy=None,
            cooldown_percent_per_strategy=None,
        ):
            if result.current_benchmark is None:
                continue

            results.append(
                {
                    "rate": args.rate,
                    "rate_type": args.rate_type,
                    "results": {**result.current_benchmark.model_dump()},
                },
            )
    except Exception as e:
        print(f"Error running benchmark: {e}")
        sys.exit(1)

    print("\nBenchmark completed successfully!")

    # Extract metrics from each result
    all_metrics = []
    for result in results:
        metrics = extract_metrics_from_result(result)
        all_metrics.append(metrics)

    # Generate summary table
    print("\n" + "=" * 80)
    print("LATENCY METRICS SUMMARY")
    print("=" * 80)
    print(format_percentile_table(all_metrics))

    # Generate aggregate statistics
    print("\n")
    print(calculate_aggregate_stats(all_metrics))


if __name__ == "__main__":
    asyncio.run(main())


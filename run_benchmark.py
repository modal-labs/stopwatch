import logging

import click
import modal


@click.command()
@click.option("--data", type=str, default="neuralmagic/LLM_compression_calibration")
@click.option("--model", type=str, required=True)
def run_benchmark(model: str, data: str):
    """Run a benchmark on Modal.

    Args:
        model (str): Name of the model to benchmark.
        data (str): Name of the dataset to use for benchmarking.
    """

    f = modal.Function.from_name("stopwatch", "run_benchmark")
    fc = f.spawn(model=model, data=data)
    logging.info(f"Benchmark running at {fc.object_id}")


if __name__ == "__main__":
    run_benchmark()

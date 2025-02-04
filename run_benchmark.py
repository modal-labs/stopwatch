import click
import modal


@click.command()
@click.option(
    "--model", type=str, required=True, help="Name of the model to benchmark."
)
@click.option(
    "--data",
    type=str,
    default="prompt_tokens=512,generated_tokens=128",
    help="The data source to use for benchmarking. Depending on the data-type, it should be a path to a data file containing prompts to run (ex: data.txt), a HuggingFace dataset name (ex: 'neuralmagic/LLM_compression_calibration'), or a configuration for emulated data (ex: 'prompt_tokens=128,generated_tokens=128').",
)
@click.option(
    "--gpu",
    type=str,
    default="H100",
    help="GPU to run the vLLM server on. Defaults to 'H100'.",
)
@click.option(
    "--vllm-docker-tag",
    type=str,
    default="latest",
    help="Docker tag to use for vLLM. Defaults to 'latest'.",
)
@click.option(
    "--vllm-env-vars",
    "-e",
    type=str,
    multiple=True,
    default=[],
    help="Environment variables to set on the vLLM server. Each argument should be in the format of 'KEY=VALUE'",
)
@click.option(
    "--vllm-extra-args",
    type=str,
    default="",
    help="Extra arguments to pass to the vLLM server.",
)
def run_benchmark(
    data: str,
    gpu: str,
    model: str,
    vllm_docker_tag: str,
    vllm_env_vars: list[str],
    vllm_extra_args: str,
):
    f = modal.Function.from_name("stopwatch", "run_benchmark")
    fc = f.spawn(
        model=model,
        data=data,
        gpu=gpu,
        vllm_docker_tag=vllm_docker_tag,
        vllm_env_vars={k: v for k, v in (e.split("=") for e in vllm_env_vars)},
        vllm_extra_args=vllm_extra_args.split(" ") if vllm_extra_args else [],
    )
    print(f"Benchmark running at {fc.object_id}")


if __name__ == "__main__":
    run_benchmark()

import typer

from .run_benchmark import run_benchmark_cli
from .run_benchmark_suite import run_benchmark_suite_cli
from .run_profiler import run_profiler_cli

app = typer.Typer()
app.command(name="run-benchmark")(run_benchmark_cli)
app.command(name="run-benchmark-suite")(run_benchmark_suite_cli)
app.command(name="run-profiler")(run_profiler_cli)


if __name__ == "__main__":
    app()

import typer

from .benchmark import benchmark_cli
from .profile import profile_cli
from .provision import provision_cli
from .provision_and_benchmark import provision_and_benchmark_cli

app = typer.Typer()
app.command(name="benchmark")(benchmark_cli)
app.command(name="profile")(profile_cli)
app.command(name="provision")(provision_cli)
app.command(name="provision-and-benchmark")(provision_and_benchmark_cli)


if __name__ == "__main__":
    app()

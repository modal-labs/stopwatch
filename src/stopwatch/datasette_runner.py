import asyncio
from collections.abc import Callable
from pathlib import Path

import modal

from .resources import db_volume, web_app

DB_PATH = "/db"


datasette_image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .uv_pip_install("datasette")
    .run_commands(
        "datasette install git+https://github.com/jackcook/stopwatch-plot.git@ff5b060",
    )
)

with datasette_image.imports():
    from datasette.app import Datasette


@web_app.cls(
    image=datasette_image,
    volumes={DB_PATH: db_volume},
)
@modal.concurrent(max_inputs=100)
class DatasetteRunner:
    """Modal class that runs a Datasette server."""

    @modal.asgi_app(label="datasette")
    def start(self) -> Callable:
        """Start the Datasette server."""
        ds = Datasette(
            files=[str(Path(DB_PATH) / "stopwatch.db")],
            settings={"sql_time_limit_ms": 10000},
            cors=True,
        )
        asyncio.run(ds.invoke_startup())
        return ds.app()

import modal

from .resources import app, datasette_volume


DATASETTE_PATH = "/datasette"


datasette_image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install("datasette", "numpy")
    .run_commands(
        "datasette install git+https://github.com/jackcook/stopwatch-plot.git"
    )
)

with datasette_image.imports():
    import asyncio
    import os

    from datasette.app import Datasette


@app.cls(
    image=datasette_image,
    volumes={DATASETTE_PATH: datasette_volume},
    allow_concurrent_inputs=100,
)
class DatasetteRunner:

    id: str = modal.parameter(default="stopwatch")

    @modal.asgi_app(label="datasette")
    def start(self):
        ds = Datasette(
            files=[os.path.join(DATASETTE_PATH, f"{self.id}.db")],
            settings={"sql_time_limit_ms": 10000},
        )
        asyncio.run(ds.invoke_startup())
        return ds.app()

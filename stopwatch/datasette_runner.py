import modal

from .resources import app, db_volume


DB_PATH = "/db"


datasette_image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install("datasette")
    .run_commands(
        "datasette install git+https://github.com/jackcook/stopwatch-plot.git@ff5b060"
    )
    .add_local_python_source("cli")
)

with datasette_image.imports():
    import asyncio
    import os

    from datasette.app import Datasette


@app.cls(
    image=datasette_image,
    volumes={DB_PATH: db_volume},
)
@modal.concurrent(max_inputs=100)
class DatasetteRunner:
    @modal.asgi_app(label="datasette")
    def start(self):
        ds = Datasette(
            files=[os.path.join(DB_PATH, "stopwatch.db")],
            settings={"sql_time_limit_ms": 10000},
            cors=True,
        )
        asyncio.run(ds.invoke_startup())
        return ds.app()

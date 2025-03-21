import modal

from .resources import app, db_volume


DB_PATH = "/db"


datasette_image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install("datasette", "numpy", "SQLAlchemy")
    .run_commands(
        "datasette install git+https://github.com/jackcook/stopwatch-plot.git@04cbbcb"
    )
)

with datasette_image.imports():
    import asyncio
    import os

    from datasette.app import Datasette


@app.cls(
    image=datasette_image,
    volumes={DB_PATH: db_volume},
    allow_concurrent_inputs=100,
)
class DatasetteRunner:
    @modal.asgi_app(label="datasette")
    def start(self):
        ds = Datasette(
            files=[os.path.join(DB_PATH, "stopwatch.db")],
            settings={"sql_time_limit_ms": 10000},
        )
        asyncio.run(ds.invoke_startup())
        return ds.app()

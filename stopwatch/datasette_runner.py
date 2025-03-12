import modal

from .resources import app, datasette_volume


DATASETTE_PATH = "/datasette"


datasette_image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install("datasette")
    .run_commands(
        "datasette install git+https://github.com/jackcook/stopwatch-plot.git@b11f92e"
    )
)

with datasette_image.imports():
    import asyncio
    import os

    from datasette.app import Datasette


def datasette_cls(
    image=datasette_image,
    volumes={DATASETTE_PATH: datasette_volume},
    allow_concurrent_inputs=100,
):
    def decorator(cls):
        return app.cls(
            image=image,
            volumes=volumes,
            allow_concurrent_inputs=allow_concurrent_inputs,
        )(cls)

    return decorator


class DatasetteRunner:
    def start(self, id):
        ds = Datasette(
            files=[os.path.join(DATASETTE_PATH, f"{id}.db")],
            settings={"sql_time_limit_ms": 10000},
        )
        asyncio.run(ds.invoke_startup())
        return ds.app()


@datasette_cls()
class MainDatasetteRunner(DatasetteRunner):
    @modal.asgi_app(label="datasette")
    def start(self):
        return super().start("stopwatch")


@datasette_cls()
class RegionDatasetteRunner(DatasetteRunner):
    @modal.asgi_app(label="datasette-region")
    def start(self):
        return super().start("region-test")

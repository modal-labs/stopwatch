from .db import engine, session
from .base import Base
from .benchmark import AveragedBenchmark, Benchmark, BenchmarkDefaults


def create_all():
    Base.metadata.create_all(engine)

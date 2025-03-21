from .db import engine, session
from .base import Base
from .benchmark import Benchmark, BenchmarkDefaults, benchmark_cls_factory


def create_all():
    Base.metadata.create_all(engine)

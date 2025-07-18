from .base import Base
from .benchmark import (
    Benchmark,
    RateType,
    benchmark_class_factory,
)
from .db import engine, session


def create_all() -> None:
    """Create base tables in the database."""
    Base.metadata.create_all(engine)


__all__ = [
    "Benchmark",
    "RateType",
    "benchmark_class_factory",
    "create_all",
    "session",
]

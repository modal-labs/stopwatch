from typing import Sequence

from guidellm.core.serializable import Serializable
from pydantic import Field


class Histogram(Serializable):

    data: Sequence[float] = Field(
        default_factory=list,
        description="The values in each bin of the histogram.",
    )

    bins: Sequence[float] = Field(
        default_factory=list,
        description="The bin edges of the histogram, including the left edge of the first bin and the right edge of the last bin.",
    )

    def __str__(self):
        return f"Histogram(data={self.data}, bins={self.bins})"

    def __len__(self):
        return len(self.data)

    def set_data(self, new_data: Sequence[float], bins: Sequence[float]):
        assert len(new_data) == len(bins) - 1
        self.data = list(new_data)
        self.bins = list(bins)

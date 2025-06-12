from collections.abc import Sequence

from guidellm.objects import StandardBaseModel
from pydantic import Field


class Histogram(StandardBaseModel):
    """A histogram of a distribution."""

    data: Sequence[float] = Field(
        default_factory=list,
        description="The values in each bin of the histogram.",
    )

    bins: Sequence[float] = Field(
        default_factory=list,
        description=(
            "The bin edges of the histogram, including the left edge of the first bin "
            "and the right edge of the last bin."
        ),
    )

    def __str__(self) -> str:
        """Return a string representation of the histogram."""
        return f"Histogram(data={self.data}, bins={self.bins})"

    def __len__(self) -> int:
        """Return the number of bins in the histogram."""
        return len(self.data)

    def set_data(self, new_data: Sequence[float], bins: Sequence[float]) -> None:
        """Set the data and bins of the histogram."""
        if len(new_data) != len(bins) - 1:
            msg = f"len({new_data}) != len({bins}) - 1"
            raise ValueError(msg)

        self.data = list(new_data)
        self.bins = list(bins)

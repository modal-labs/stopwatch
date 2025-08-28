from collections.abc import Iterator
from typing import Any

from guidellm.request import GenerationRequest, GenerativeRequestLoader


class CustomGenerativeRequestLoader(GenerativeRequestLoader):
    """
    A wrapper around GenerativeRequestLoader that allows for modifications to
    be made to GuideLLM requests.

    These are both useful when testing structured outputs, e.g.
    https://docs.vllm.ai/en/latest/features/structured_outputs.html
    """

    def __init__(
        self,
        extra_body: dict[str, Any] | None = None,
        *,
        use_chat_completions: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Create a custom generative request loader.

        :param: extra_body: Extra parameters to add to the body of each request.
        :param: use_chat_completions: Whether to use the chat completions endpoint,
            as opposed to the text completions endpoint.
        :param: kwargs: Additional keyword arguments to pass to the
            GenerativeRequestLoader constructor.
        """

        super().__init__(**kwargs)
        self.extra_body = extra_body or {}
        self.use_chat_completions = use_chat_completions

    def __iter__(self) -> Iterator[GenerationRequest]:
        """Iterate over the requests in the loader."""

        for item in super().__iter__():
            for k, v in self.extra_body.items():
                item.params[k] = v

            if self.use_chat_completions:
                item.request_type = "chat_completions"

            yield item

    def __len__(self) -> int:
        """Return the number of unique requests in the loader."""
        return super().__len__()

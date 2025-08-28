import httpx
from guidellm.backend.openai import OpenAIHTTPBackend


class CustomOpenAIHTTPBackend(OpenAIHTTPBackend):
    """A custom OpenAI HTTP backend that increases the number of maximum redirects."""

    def _get_async_client(self) -> httpx.AsyncClient:
        if self._async_client is None or self._async_client.is_closed:
            client = super()._get_async_client()
            client.max_redirects = 1000
            self._async_client = client

        return self._async_client

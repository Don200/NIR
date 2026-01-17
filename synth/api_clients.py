import functools
import logging
import time
from typing import List, Optional
from urllib.parse import urlparse

import httpx
import requests
import urllib3
from openai import APITimeoutError, OpenAI


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def retry_on_timeout():
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            retries = self.retries
            delay = self.delay
            attempt = 0
            while attempt < retries:
                try:
                    return func(self, *args, **kwargs)
                except APITimeoutError as exc:
                    attempt += 1
                    logger.warning("Attempt %d/%d: %r", attempt, retries, exc)
                    if attempt == retries:
                        logger.error("Retry limit reached.")
                        raise
                    time.sleep(delay)

        return wrapper

    return decorator


def ensure_embeddings_endpoint(api_base: Optional[str]) -> str:
    if not api_base:
        raise RuntimeError("embeddings api_base must be set for EmbedderClient.")
    if api_base.rstrip("/").endswith("/embeddings"):
        return api_base.rstrip("/")
    return f"{api_base.rstrip('/')}/embeddings"


def normalize_base_url(api_base: Optional[str]) -> str:
    if not api_base:
        raise RuntimeError("api_base must be set.")
    cleaned = api_base.rstrip("/")
    if cleaned.endswith("/embeddings"):
        return cleaned[: -len("/embeddings")]
    return cleaned


class OpenAILLMClient:
    def __init__(
        self,
        *,
        llm_url: str,
        api_key: str,
        path2llm: str,
        retries_count: int = 2,
        delay: int = 20,
    ):
        self.client = OpenAI(
            base_url=llm_url,
            api_key=api_key,
            http_client=httpx.Client(verify=False),
        )
        self.api_key = api_key
        self.model = path2llm
        self.retries = retries_count
        self.delay = delay

    @retry_on_timeout()
    def ask(
        self,
        *,
        messages: List[dict],
        temperature: float = 0.1,
        top_p: float = 0.95,
        timeout: float = 90,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
        )
        return response.choices[0].message.content or ""


class EmbedderClient:
    def __init__(
        self,
        *,
        embeddings_endpoint: str,
        model: str,
        timeout: int,
        api_key: Optional[str] = None,
        tokenize_url: Optional[str] = None,
    ):
        self.embeddings_endpoint = embeddings_endpoint
        self.model = model
        self.timeout = timeout
        self.api_key = api_key

        parsed = urlparse(embeddings_endpoint)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        self.tokenize_url = tokenize_url if tokenize_url else f"{base_url}/tokenize"

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.post(
            self.embeddings_endpoint,
            headers=headers,
            json={"model": self.model, "input": texts},
            verify=False,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data["data"]]

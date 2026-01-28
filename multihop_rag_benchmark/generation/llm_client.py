"""OpenAI-compatible LLM and Embedding clients."""

from typing import List, Optional, Dict, Any
import time
import logging

logger = logging.getLogger(__name__)


class LLMClient:
    """OpenAI-compatible LLM client."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        api_base: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

        client_kwargs = {"api_key": api_key, "timeout": timeout}
        if api_base:
            client_kwargs["base_url"] = api_base

        self.client = OpenAI(**client_kwargs)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate response from LLM."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Chat completion with messages."""
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=tokens,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"LLM request failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise

    def generate_with_context(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate answer given query and retrieved context."""
        if system_prompt is None:
            system_prompt = (
                "You are a helpful assistant. Answer the question based only on the "
                "provided context. If the context doesn't contain enough information "
                "to answer, say so. Be concise and precise."
            )

        prompt = f"""Context:
{context}

Question: {query}

Answer:"""

        return self.generate(prompt, system_prompt=system_prompt)


class EmbeddingClient:
    """OpenAI-compatible embedding client."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-ada-002",
        api_base: Optional[str] = None,
        batch_size: int = 100,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries

        client_kwargs = {"api_key": api_key, "timeout": timeout}
        if api_base:
            client_kwargs["base_url"] = api_base

        self.client = OpenAI(**client_kwargs)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts."""
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a single batch."""
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts,
                )
                # Sort by index to maintain order
                sorted_data = sorted(response.data, key=lambda x: x.index)
                return [item.embedding for item in sorted_data]
            except Exception as e:
                logger.warning(f"Embedding request failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

    def embed_single(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        return self.embed([text])[0]

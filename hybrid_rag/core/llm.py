"""LLM client for OpenAI-compatible APIs."""

from typing import Optional

from langfuse.openai import OpenAI
from langfuse.decorators import observe

from .config import LLMConfig
from .prompts import SYSTEM_PROMPT_RU, FEW_SHOT_EXAMPLES_RU, build_few_shot_prompt


class LLMClient:
    """OpenAI-compatible LLM client."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    @observe(as_type="generation")
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

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.get_max_tokens(),
        )

        return response.choices[0].message.content or ""

    @observe(as_type="generation")
    def generate_with_context(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate answer using retrieved context with few-shot examples."""
        prompt = build_few_shot_prompt(
            context=context,
            question=query,
            examples=FEW_SHOT_EXAMPLES_RU,
        )

        return self.generate(
            prompt=prompt,
            system_prompt=system_prompt or SYSTEM_PROMPT_RU,
        )

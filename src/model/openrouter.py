"""OpenRouter client for the AICrypto CTF agent.

Works with any model on OpenRouter (free or paid) through the OpenAI-compatible
endpoint.  Select it in config/model.yaml with ``client: "openrouter"``; the
``deployment`` field is the OpenRouter model slug (copy it from the model's page).
"""

import os
import time
from typing import Any, Dict

import dotenv
import openai

from .base_model import BaseModel

dotenv.load_dotenv()


class OpenRouterModel(BaseModel):
    def __init__(self, model_config: Dict[str, Any], system_prompt: str = None):
        super().__init__(model_config)
        self.system_prompt = system_prompt
        self.max_tokens = int(model_config.get("max_tokens", 8192))
        self.reasoning_effort = model_config.get("reasoning_effort")  # optional
        self.init_client()
        self.init_prompt()

    def init_client(self):
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set (put it in .env)")
        self.client = openai.OpenAI(
            api_key=key,
            base_url="https://openrouter.ai/api/v1",
            max_retries=0,  # we do our own backoff below
        )

    def init_prompt(self):
        self.messages = (
            [{"role": "system", "content": self.system_prompt}]
            if self.system_prompt
            else []
        )

    def get_response(self) -> Dict[str, str]:
        kwargs: Dict[str, Any] = dict(
            model=self.deployment,
            messages=self.messages,
            max_tokens=self.max_tokens,
        )
        if self.reasoning_effort:
            kwargs["extra_body"] = {"reasoning": {"effort": self.reasoning_effort}}

        # Free tiers rate-limit hard; back off on 429 / transient 5xx.
        last_exc = None
        for attempt in range(6):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                msg = resp.choices[0].message
                answer = msg.content or ""
                thinking = getattr(msg, "reasoning", None) or ""
                return {"thinking": thinking, "answer": answer}
            except openai.RateLimitError as exc:
                last_exc = exc
                time.sleep(min(60, 5 * 2 ** attempt))
            except openai.APIStatusError as exc:
                if exc.status_code >= 500:
                    last_exc = exc
                    time.sleep(min(60, 5 * 2 ** attempt))
                else:
                    raise
            except (openai.APIConnectionError, openai.APITimeoutError) as exc:
                last_exc = exc
                time.sleep(min(60, 5 * 2 ** attempt))
        raise last_exc

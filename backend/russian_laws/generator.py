"""Модуль генератора ответов на основе OpenAI API."""

import openai
from omegaconf import DictConfig


class LLMGenerator:
    """Генератор ответов с использованием OpenAI API (через роутер)."""

    def __init__(self, config: DictConfig):
        """Инициализация генератора.

        Args:
            config: Конфигурация Hydra с параметрами генератора
        """
        self.config = config
        self.model_name = config.generator.model
        self.system_prompt = config.generator.system_prompt
        self.max_tokens = getattr(config.generator, "max_tokens", 512)
        self.temperature = getattr(config.generator, "temperature", 0.7)

        # Инициализация OpenAI клиента
        self.client = openai.OpenAI(
            api_key=config.generator.api_key,
            base_url=config.generator.base_url,
        )
        print(f"LLM генератор инициализирован (модель: {self.model_name})")

    def generate(self, query: str, context: str) -> str:
        """Генерирует ответ на основе запроса и контекста.

        Args:
            query: Вопрос пользователя
            context: Контекст из релевантных статей законов

        Returns:
            Сгенерированный ответ
        """
        user_prompt = self._build_user_prompt(query, context)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        return response.choices[0].message.content

    def _build_user_prompt(self, query: str, context: str) -> str:
        """Формирует промпт для LLM.

        Args:
            query: Вопрос пользователя
            context: Контекст из статей законов

        Returns:
            Сформированный промпт
        """
        return f"""Контекст (релевантные статьи законов):
{context}

---

Вопрос пользователя: {query}

Дай точный и структурированный ответ на основе предоставленного контекста."""

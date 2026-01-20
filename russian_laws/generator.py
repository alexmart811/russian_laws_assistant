"""Модуль генератора ответов на основе локальной модели HuggingFace."""

from omegaconf import DictConfig
from transformers import pipeline


class LLMGenerator:
    """Генератор ответов с использованием локальной модели HuggingFace."""

    def __init__(self, config: DictConfig):
        """Инициализация генератора.

        Args:
            config: Конфигурация Hydra с параметрами генератора
        """
        self.config = config
        self.model_name = config.generator.model
        self.system_prompt = config.generator.system_prompt
        self.max_new_tokens = getattr(config.generator, "max_new_tokens", 512)
        self.device = getattr(config.generator, "device", "cuda")

        # Инициализация pipeline
        print(f"Загрузка модели {self.model_name}...")
        self.pipe = pipeline(
            "text-generation",
            model=self.model_name,
            device=self.device,
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

        result = self.pipe(messages, max_new_tokens=self.max_new_tokens)

        # Извлекаем ответ ассистента
        generated = result[0]["generated_text"]
        # generated - это список сообщений, берем последнее (ответ ассистента)
        if isinstance(generated, list) and len(generated) > 0:
            last_message = generated[-1]
            if isinstance(last_message, dict) and "content" in last_message:
                return last_message["content"]
        return str(generated)

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

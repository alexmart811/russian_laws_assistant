"""Модуль генератора ответов на основе OpenAI API."""

import json
import re

import openai
from omegaconf import DictConfig

_NO_INFO_ANSWER = (
    "В предоставленных документах информация по этому вопросу отсутствует."
)


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

        self.client = openai.OpenAI(
            api_key=config.generator.api_key,
            base_url=config.generator.base_url,
        )

        self.answer_format_instructions = {
            "number": (
                "\n\nВАЖНО: Ответь ТОЛЬКО числом (целым или дробным). "
                "Без пояснений и дополнительного текста. "
                "Если информации нет в контексте, верни: null"
            ),
            "boolean": (
                "\n\nВАЖНО: Ответь ТОЛЬКО 'true' или 'false'. "
                "Без пояснений и дополнительного текста. "
                "Если информации нет в контексте, верни: null"
            ),
            "date": (
                "\n\nВАЖНО: Ответь ТОЛЬКО датой в формате ISO 8601 (YYYY-MM-DD). "
                "Без пояснений и дополнительного текста. "
                "Если информации нет в контексте, верни: null"
            ),
            "name": (
                "\n\nВАЖНО: Ответь ТОЛЬКО запрашиваемым именем/названием. "
                "Без пояснений и дополнительного текста. "
                "Если информации нет в контексте, верни: null"
            ),
            "names": (
                '\n\nВАЖНО: Ответь ТОЛЬКО JSON массивом имён: ["имя1", "имя2"]. '
                "Без пояснений и дополнительного текста. "
                "Если информации нет в контексте, верни: null"
            ),
            "free_text": (
                "\n\nДай исчерпывающий ответ (1-3 абзаца, не более 280 символов). "
                f'Если информации нет в контексте, верни: "{_NO_INFO_ANSWER}"'
            ),
        }

        print(f"LLM генератор инициализирован (модель: {self.model_name})")

    def generate(
        self,
        query: str,
        context: str,
        answer_type: str = "free_text",
        return_usage: bool = False,
    ) -> str | tuple[str, dict]:
        """Генерирует ответ на основе запроса и контекста.

        Args:
            query: Вопрос пользователя
            context: Контекст из релевантных статей законов
            answer_type: Тип ожидаемого ответа
            return_usage: Если True, возвращает кортеж (ответ, usage_info)

        Returns:
            Сгенерированный ответ или кортеж (ответ, usage_info)
        """
        user_prompt = self._build_user_prompt(query, context, answer_type)

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

        answer = response.choices[0].message.content
        processed_answer = self._postprocess_answer(answer, answer_type)

        if return_usage:
            usage_info = {
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": (
                    response.usage.completion_tokens if response.usage else 0
                ),
            }
            return processed_answer, usage_info

        return processed_answer

    def _build_user_prompt(
        self, query: str, context: str, answer_type: str = "free_text"
    ) -> str:
        """Формирует промпт для LLM."""
        format_instruction = self.answer_format_instructions.get(
            answer_type, self.answer_format_instructions["free_text"]
        )

        return (
            f"Контекст (релевантные статьи законов):\n{context}\n\n---\n\n"
            f"Вопрос пользователя: {query}\n{format_instruction}\n\n"
            "Дай точный и структурированный ответ на основе предоставленного контекста."
        )

    def _postprocess_answer(self, answer: str, answer_type: str) -> str:
        """Постобработка и валидация ответа."""
        answer = answer.strip()

        if answer.lower() == "null" or answer == _NO_INFO_ANSWER:
            return answer

        if answer_type == "number":
            match = re.search(r"-?\d+\.?\d*", answer)
            return match.group(0) if match else answer

        if answer_type == "boolean":
            answer_lower = answer.lower()
            if any(w in answer_lower for w in ("true", "yes", "да")):
                return "true"
            if any(w in answer_lower for w in ("false", "no", "нет")):
                return "false"
            return answer

        if answer_type == "date":
            match = re.search(r"\d{4}-\d{2}-\d{2}", answer)
            return match.group(0) if match else answer

        if answer_type == "name":
            answer = re.sub(
                r"^(?:(?:The|Это)\s+)?(?:name|entity|person|название|имя)\s+(?:is|—|:)\s+",
                "",
                answer,
                flags=re.IGNORECASE,
            )
            return answer.strip()

        if answer_type == "names":
            json_match = re.search(r"\[.*?\]", answer, re.DOTALL)
            if json_match:
                try:
                    names = json.loads(json_match.group(0))
                    return json.dumps(names, ensure_ascii=False)
                except (json.JSONDecodeError, TypeError):
                    pass
            return answer

        return answer

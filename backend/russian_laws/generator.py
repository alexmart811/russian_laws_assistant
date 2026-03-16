"""Модуль генератора ответов на основе OpenAI API."""

import re

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

        # Инструкции по форматам ответов для разных типов
        self.answer_format_instructions = {
            "number": "\n\nIMPORTANT: Answer with ONLY a number (int or float). No explanation or additional text. If the information is not available in the provided context, return exactly: null",
            "boolean": "\n\nIMPORTANT: Answer with ONLY 'true' or 'false'. No explanation or additional text. If the information is not available in the provided context, return exactly: null",
            "date": "\n\nIMPORTANT: Answer with ONLY a date in ISO 8601 format (YYYY-MM-DD). No explanation or additional text. If the information is not available in the provided context, return exactly: null",
            "name": "\n\nIMPORTANT: Answer with ONLY the name/entity requested. No explanation or additional text. If the information is not available in the provided context, return exactly: null",
            "names": '\n\nIMPORTANT: Answer with ONLY a JSON array of names: ["name1", "name2"]. No explanation or additional text. If the information is not available in the provided context, return exactly: null',
            "free_text": '\n\nProvide a comprehensive answer (1-3 paragraphs, maximum 280 characters). If the information is not available in the provided context, return exactly: "There is no information on this question in the provided documents."',
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
            answer_type: Тип ожидаемого ответа (number, boolean, date, name, names, free_text)
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

        # Постобработка для структурированных ответов
        processed_answer = self._postprocess_answer(answer, answer_type)

        if return_usage:
            usage_info = {
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens
                if response.usage
                else 0,
            }
            return processed_answer, usage_info

        return processed_answer

    def _build_user_prompt(
        self, query: str, context: str, answer_type: str = "free_text"
    ) -> str:
        """Формирует промпт для LLM.

        Args:
            query: Вопрос пользователя
            context: Контекст из статей законов
            answer_type: Тип ожидаемого ответа

        Returns:
            Сформированный промпт
        """
        # Добавляем инструкцию по формату ответа
        format_instruction = self.answer_format_instructions.get(
            answer_type, self.answer_format_instructions["free_text"]
        )

        return f"""Контекст (релевантные статьи законов):
{context}

---

Вопрос пользователя: {query}
{format_instruction}

Дай точный и структурированный ответ на основе предоставленного контекста."""

    def _postprocess_answer(self, answer: str, answer_type: str) -> str:
        """Постобработка и валидация ответа.

        Args:
            answer: Сырой ответ от LLM
            answer_type: Ожидаемый тип ответа

        Returns:
            Очищенный и валидированный ответ
        """
        answer = answer.strip()

        # Если ответ указывает на отсутствие информации, возвращаем как есть
        if (
            answer.lower() == "null"
            or answer
            == "There is no information on this question in the provided documents."
        ):
            return answer

        if answer_type == "number":
            # Извлекаем первое число
            match = re.search(r"-?\d+\.?\d*", answer)
            return match.group(0) if match else answer

        elif answer_type == "boolean":
            # Нормализуем к true/false
            answer_lower = answer.lower()
            if "true" in answer_lower or "yes" in answer_lower:
                return "true"
            elif "false" in answer_lower or "no" in answer_lower:
                return "false"
            return answer

        elif answer_type == "date":
            # Извлекаем дату в формате YYYY-MM-DD
            match = re.search(r"\d{4}-\d{2}-\d{2}", answer)
            return match.group(0) if match else answer

        elif answer_type == "name":
            # Убираем лишние слова типа "The name is", оставляем только имя
            answer = re.sub(
                r"^(?:The\s+)?(?:name|entity|person)\s+is\s+",
                "",
                answer,
                flags=re.IGNORECASE,
            )
            return answer.strip()

        elif answer_type == "names":
            # Пробуем извлечь JSON массив или список через запятую
            try:
                import json

                # Ищем JSON массив в тексте
                json_match = re.search(r"\[.*?\]", answer, re.DOTALL)
                if json_match:
                    names = json.loads(json_match.group(0))
                    return json.dumps(names)
            except:
                pass
            return answer

        # Для free_text возвращаем как есть
        return answer

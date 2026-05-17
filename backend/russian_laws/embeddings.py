import numpy as np
import torch
from omegaconf import DictConfig
from transformers import AutoModel, AutoTokenizer

# E5-семейство требует instruction prefix
_E5_MODEL_PREFIXES = ("intfloat/e5-", "intfloat/multilingual-e5-")
# Jina v3 использует trust_remote_code и task-специфичные LoRA адаптеры
_JINA_V3_PREFIX = "jinaai/jina-embeddings-v3"
_TRUST_REMOTE_CODE_PREFIXES = (_JINA_V3_PREFIX,)


class EmbeddingModel:
    def __init__(self, config: DictConfig):
        self.config = config
        self.model_name = config.embedding.model_name
        self.device = torch.device(config.embedding.device)
        self.batch_size = config.embedding.batch_size
        self.max_length = config.embedding.max_length
        self.normalize = config.embedding.normalize
        self.pooling = config.embedding.pooling
        self._requires_prefix = any(
            self.model_name.startswith(p) for p in _E5_MODEL_PREFIXES
        ) or config.embedding.get("force_e5_prefix", False)
        self._is_jina_v3 = self.model_name.startswith(_JINA_V3_PREFIX)
        trust_remote_code = any(
            self.model_name.startswith(p) for p in _TRUST_REMOTE_CODE_PREFIXES
        ) or bool(config.embedding.get("trust_remote_code", False))
        # Matryoshka: усечение размерности (поддерживает Jina v3)
        self.truncate_dim = config.embedding.get("truncate_dim", None)

        print(f"Загрузка модели эмбеддингов: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=trust_remote_code
        )
        self.model = AutoModel.from_pretrained(
            self.model_name, trust_remote_code=trust_remote_code
        ).to(self.device)
        self.model.eval()
        print(f"Модель загружена на устройство: {self.device}")

    def _add_prefix(self, texts: list[str], prefix: str) -> list[str]:
        if not self._requires_prefix:
            return texts
        return [f"{prefix}{t}" for t in texts]

    def _mean_pooling(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def _cls_pooling(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        return token_embeddings[:, 0, :]

    def _max_pooling(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        token_embeddings[input_mask_expanded == 0] = -1e9
        return torch.max(token_embeddings, 1)[0]

    @torch.no_grad()
    def _encode_raw(self, texts: list[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**encoded)
        token_embeddings = outputs.last_hidden_state

        if self.pooling == "mean":
            embeddings = self._mean_pooling(token_embeddings, encoded["attention_mask"])
        elif self.pooling == "cls":
            embeddings = self._cls_pooling(token_embeddings)
        elif self.pooling == "max":
            embeddings = self._max_pooling(token_embeddings, encoded["attention_mask"])
        else:
            raise ValueError(f"Неизвестный тип pooling: {self.pooling}")

        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    @torch.no_grad()
    def _encode_jina(self, texts: list[str], task: str) -> torch.Tensor:
        encode_kwargs: dict = {
            "task": task,
            "max_length": self.max_length,
            "batch_size": max(1, len(texts)),
        }
        if self.truncate_dim:
            encode_kwargs["truncate_dim"] = int(self.truncate_dim)

        embeddings = self.model.encode(texts, **encode_kwargs)
        embeddings = torch.from_numpy(np.asarray(embeddings)).to(self.device).float()

        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

    def encode(self, texts: list[str]) -> torch.Tensor:
        """Кодирует запросы (query prefix для E5, retrieval.query task для Jina v3)."""
        if self._is_jina_v3:
            return self._encode_jina(texts, task="retrieval.query")
        return self._encode_raw(self._add_prefix(texts, "query: "))

    def encode_passages(self, texts: list[str]) -> torch.Tensor:
        """Кодирует документы/пассажи."""
        if self._is_jina_v3:
            return self._encode_jina(texts, task="retrieval.passage")
        return self._encode_raw(self._add_prefix(texts, "passage: "))

    def encode_batch(
        self, texts: list[str], is_passage: bool = False
    ) -> list[list[float]]:
        all_embeddings = []
        encode_fn = self.encode_passages if is_passage else self.encode

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            embeddings = encode_fn(batch)
            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0).tolist()

    def get_embedding_dim(self) -> int:
        if self.truncate_dim:
            return int(self.truncate_dim)
        return self.model.config.hidden_size

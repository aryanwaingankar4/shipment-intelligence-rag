"""Bi-encoder (embeddings) and cross-encoder (reranker) wrappers.

Heavy imports happen lazily so unit tests and the CLI help don't load torch.
"""

from functools import cached_property


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name

    @cached_property
    def _model(self):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.model_name)

    def encode(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        # normalized so cosine similarity == dot product
        vectors = self._model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.tolist()

    def warm_up(self) -> None:
        self.encode(["warm up"])


class Reranker:
    def __init__(self, model_name: str):
        self.model_name = model_name

    @cached_property
    def _model(self):
        from sentence_transformers import CrossEncoder

        return CrossEncoder(self.model_name)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return [float(s) for s in self._model.predict([[query, t] for t in texts])]

    def warm_up(self) -> None:
        self.score("warm up", ["warm up"])

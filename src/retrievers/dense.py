import json
import hashlib
import os
from pathlib import Path

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

# Keep Transformers from importing TensorFlow/Keras when this project only needs
# PyTorch sentence-transformer inference.
os.environ.setdefault("USE_TF", "0")

from src.retrievers.base import BaseRetriever


class DenseRetriever(BaseRetriever):
    def __init__(
        self,
        corpus_path: str,
        model_name: str = "intfloat/multilingual-e5-base",
        embedding_dir: str = "embeddings",
        backend: str = "auto",
        device: str | None = None,
    ):
        super().__init__(corpus_path)
        self.model_name = model_name
        self.backend = backend
        self.device = self._resolve_device(device)
        self.embedding_dir = Path(embedding_dir)
        self.embedding_dir.mkdir(parents=True, exist_ok=True)

        self.corpus = self._load_corpus()
        self.doc_ids = [item["doc_id"] for item in self.corpus]
        self.doc_texts = [item["text"] for item in self.corpus]
        self.index = None
        self.model = None
        self._query_embedding_cache = {}

        self._initialize_backend()
        cache_key = self._cache_key()
        self.index_path = self.embedding_dir / f"{cache_key}.faiss.index"
        self.embedding_path = self.embedding_dir / f"{cache_key}.doc_embeddings.npy"
        self.doc_ids_path = self.embedding_dir / f"{cache_key}.doc_ids.npy"
        self._build_or_load_index()

    def _initialize_backend(self) -> None:
        if self.backend not in {"auto", "sentence_transformers", "lexical"}:
            raise ValueError("Dense backend must be one of: auto, sentence_transformers, lexical")
        if self.backend == "lexical":
            return
        SentenceTransformer, import_error = self._load_sentence_transformer()
        if SentenceTransformer is None or faiss is None:
            if self.backend == "sentence_transformers":
                raise ImportError(
                    "sentence-transformers and faiss-cpu are required for sentence_transformers backend."
                ) from import_error
            if import_error is not None:
                print(f"Falling back to lexical dense backend: {import_error}")
            self.backend = "lexical"
            return
        try:
            kwargs = {"device": self.device} if self.device else {}
            self.model = SentenceTransformer(self.model_name, **kwargs)
            self.backend = "sentence_transformers"
        except Exception as exc:
            if self.backend == "sentence_transformers":
                raise exc
            print(f"Falling back to lexical dense backend: {exc}")
            self.backend = "lexical"

    def _resolve_device(self, device: str | None) -> str | None:
        if not device or device == "auto":
            return device
        if device != "mps":
            return device
        try:
            import torch
        except Exception:
            print("Requested MPS device, but torch is unavailable. Falling back to default device.")
            return None
        if torch.backends.mps.is_available():
            return "mps"
        print("Requested MPS device, but MPS is unavailable. Falling back to default device.")
        return None

    def _load_sentence_transformer(self):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            return None, exc
        return SentenceTransformer, None

    def _cache_key(self) -> str:
        raw_key = f"{self.backend}:{self.model_name}:{self.corpus_path}:{self._corpus_fingerprint()}"
        digest = hashlib.md5(raw_key.encode("utf-8")).hexdigest()[:12]
        model_slug = "".join(char if char.isalnum() else "_" for char in self.model_name.lower())
        return f"dense_{model_slug}_{digest}"

    def _corpus_fingerprint(self) -> str:
        path = Path(self.corpus_path)
        digest = hashlib.sha256()
        digest.update(str(path.resolve()).encode("utf-8"))
        with path.open("rb") as fin:
            for chunk in iter(lambda: fin.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()[:16]

    def _load_corpus(self) -> list[dict]:
        path = Path(self.corpus_path)
        if not path.exists():
            raise FileNotFoundError(f"Corpus file not found: {path}")

        corpus = []
        with path.open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                corpus.append({"doc_id": item["doc_id"], "text": item["text"]})

        if not corpus:
            raise ValueError("Corpus file is empty.")
        return corpus

    def _is_e5_model(self) -> bool:
        return "e5" in self.model_name.lower()

    def _prepare_passage(self, text: str) -> str:
        return f"passage: {text}" if self._is_e5_model() else text

    def _prepare_query(self, query: str) -> str:
        return f"query: {query}" if self._is_e5_model() else query

    def _build_or_load_index(self):
        if self.backend == "lexical":
            self.doc_embeddings = self._normalize_embeddings(np.vstack([self._lexical_embed(text) for text in self.doc_texts]))
            return

        if self.index_path.exists() and self.embedding_path.exists() and self.doc_ids_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                self.doc_embeddings = np.load(self.embedding_path)
                cached_doc_ids = np.load(self.doc_ids_path, allow_pickle=True).tolist()
                if cached_doc_ids == self.doc_ids and len(self.doc_embeddings) == len(self.doc_texts):
                    return
            except Exception:
                pass
        self._build_index()

    def _build_index(self):
        passages = [self._prepare_passage(text) for text in self.doc_texts]
        embeddings = self.model.encode(passages, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
        embeddings = self._normalize_embeddings(embeddings)

        self.doc_embeddings = embeddings.astype(np.float32)
        self.index = faiss.IndexFlatIP(self.doc_embeddings.shape[1])
        self.index.add(self.doc_embeddings)

        faiss.write_index(self.index, str(self.index_path))
        np.save(self.embedding_path, self.doc_embeddings)
        np.save(self.doc_ids_path, np.array(self.doc_ids, dtype=object))

    def _lexical_embed(self, text: str, dim: int = 384) -> np.ndarray:
        # TODO: Use a real dense encoder for final experiments; this fallback keeps the pipeline runnable.
        vector = np.zeros(dim, dtype=np.float32)
        normalized = "".join(text.lower().split())
        features = set()
        for n in (2, 3):
            for idx in range(max(0, len(normalized) - n + 1)):
                features.add(normalized[idx : idx + n])
        features.update(text.lower().split())
        for feature in features:
            digest = hashlib.md5(feature.encode("utf-8")).hexdigest()
            bucket = int(digest[:8], 16) % dim
            sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
            vector[bucket] += sign
        return vector

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        query = query or ""
        if not query.strip():
            return []

        if self.backend == "lexical":
            query_embedding = self._normalize_embeddings(np.asarray([self._lexical_embed(query)], dtype=np.float32))
            scores = np.dot(self.doc_embeddings, query_embedding[0])
            ranked_indices = np.argsort(-scores)[:top_k]
            return [
                {
                    "doc_id": self.doc_ids[idx],
                    "score": float(scores[idx]),
                    "text": self.doc_texts[idx],
                }
                for idx in ranked_indices
            ]

        query_embedding = self._encode_query(query)

        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.doc_ids):
                continue
            results.append(
                {
                    "doc_id": self.doc_ids[idx],
                    "score": float(score),
                    "text": self.doc_texts[idx],
                }
            )
        return results

    def _encode_query(self, query: str) -> np.ndarray:
        cached = self._query_embedding_cache.get(query)
        if cached is not None:
            return cached

        query_text = self._prepare_query(query)
        query_embedding = self.model.encode([query_text], convert_to_numpy=True)
        query_embedding = self._normalize_embeddings(query_embedding.astype(np.float32))
        if len(self._query_embedding_cache) < 50000:
            self._query_embedding_cache[query] = query_embedding
        return query_embedding

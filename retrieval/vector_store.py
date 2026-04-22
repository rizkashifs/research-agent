"""
Vector store backed by ChromaDB with a pure-Python TF-IDF embedding function
that requires no model downloads and works fully offline.
"""
import hashlib
import math
import re
from collections import Counter
from typing import Callable

import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings

from config import CHROMA_DB_PATH, CHROMA_COLLECTION


class TFIDFEmbeddingFunction(EmbeddingFunction):
    """
    Deterministic bag-of-words TF-IDF embedding.
    Fixed vocabulary of the 4096 most common English-ish character trigrams so
    every call is stateless and reproducible without any network access.
    """

    DIM = 512  # embedding dimension (hash buckets)

    def __call__(self, input: Documents) -> Embeddings:
        result = []
        for text in input:
            result.append(self._embed(text))
        return result

    # ------------------------------------------------------------------
    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        tokens = re.findall(r"[a-z]+", text)
        # Add character bigrams for better sub-word similarity
        bigrams = []
        for tok in tokens:
            for i in range(len(tok) - 1):
                bigrams.append(tok[i : i + 2])
        return tokens + bigrams

    def _embed(self, text: str) -> list[float]:
        tokens = self._tokenize(text)
        if not tokens:
            return [0.0] * self.DIM
        counts = Counter(tokens)
        vec = [0.0] * self.DIM
        for tok, cnt in counts.items():
            # Hash the token into a bucket
            bucket = int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.DIM
            # TF weight
            vec[bucket] += 1 + math.log(cnt)
        # L2 normalise
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.ef = TFIDFEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, content: str, topic: str, session_id: str = ""):
        doc_id = hashlib.md5(content.encode()).hexdigest()
        self.collection.upsert(
            ids=[doc_id],
            documents=[content],
            metadatas=[{"topic": topic, "session_id": session_id}],
        )

    def query(self, query: str, n_results: int = 3) -> list[dict]:
        count = self.collection.count()
        if count == 0:
            return []
        results = self.collection.query(
            query_texts=[query],
            n_results=min(n_results, count),
        )
        if not results["documents"] or not results["documents"][0]:
            return []
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]
        return [
            {"content": d, "topic": m.get("topic", ""), "distance": dist}
            for d, m, dist in zip(docs, metas, distances)
        ]

    def count(self) -> int:
        return self.collection.count()

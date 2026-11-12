"""CPU-only BM25, low-rank lexical embeddings and rank-fusion retrieval.

The low-rank vectors come from SVD of a small term-document matrix, not a
pretrained semantic encoder. No private or external corpus is required.
"""
from collections import Counter
from dataclasses import dataclass
from math import log
import re

import numpy as np


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


@dataclass(frozen=True)
class Document:
    id: str
    text: str


class BM25:
    def __init__(self, documents: list[Document], k1: float = 1.2, b: float = .75):
        if not documents or k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("nonempty documents, k1 > 0, and b in [0, 1] required")
        if len({doc.id for doc in documents}) != len(documents):
            raise ValueError("document IDs must be unique")
        self.documents = list(documents)
        self.k1, self.b = k1, b
        self.counts = [Counter(tokenize(doc.text)) for doc in documents]
        self.lengths = [sum(row.values()) for row in self.counts]
        self.avg_length = sum(self.lengths) / len(documents) if sum(self.lengths) else 1.0
        df = Counter(term for row in self.counts for term in row)
        self.idf = {term: log(1 + (len(documents) - freq + .5) / (freq + .5)) for term, freq in df.items()}

    def scores(self, query: str) -> np.ndarray:
        result = np.zeros(len(self.documents), dtype=float)
        for term in set(tokenize(query)):
            idf = self.idf.get(term, 0)
            if idf <= 0:
                continue
            for i, count in enumerate(self.counts):
                tf = count.get(term, 0)
                if tf:
                    denom = tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_length)
                    result[i] += idf * tf * (self.k1 + 1) / denom
        return result


class LatentSearch:
    """Truncated-SVD embeddings over TF-IDF (not a language model)."""

    def __init__(self, documents: list[Document], dimensions: int = 6):
        if not documents or dimensions <= 0:
            raise ValueError("nonempty documents and positive dimensions required")
        self.documents = list(documents)
        self.vocab = sorted({term for doc in documents for term in tokenize(doc.text)})
        if not self.vocab:
            raise ValueError("documents must contain at least one token")
        self.term_index = {term: index for index, term in enumerate(self.vocab)}
        matrix = np.zeros((len(documents), len(self.vocab)))
        for row, doc in enumerate(documents):
            for term, count in Counter(tokenize(doc.text)).items():
                matrix[row, self.term_index[term]] = 1 + np.log(count)
        df = np.count_nonzero(matrix, axis=0)
        self.idf = np.log((len(documents) + 1) / (df + 1)) + 1
        self.matrix = matrix * self.idf
        u, singular, vt = np.linalg.svd(self.matrix, full_matrices=False)
        rank = min(dimensions, len(singular))
        self.components = vt[:rank]
        self.doc_vectors = u[:, :rank] * singular[:rank]

    def scores(self, query: str) -> np.ndarray:
        terms = Counter(tokenize(query))
        vector = np.zeros(len(self.vocab))
        for term, count in terms.items():
            if term in self.term_index:
                index = self.term_index[term]
                vector[index] = (1 + np.log(count)) * self.idf[index]
        latent = vector @ self.components.T
        query_norm = np.linalg.norm(latent)
        row_norms = np.linalg.norm(self.doc_vectors, axis=1)
        return self.doc_vectors @ latent / np.maximum(row_norms * query_norm, 1e-12)


def order(scores: np.ndarray, documents: list[Document]) -> list[str]:
    if len(scores) != len(documents):
        raise ValueError("scores and documents do not align")
    return [doc.id for index, doc in sorted(enumerate(documents), key=lambda p: (-float(scores[p[0]]), p[1].id))]


def reciprocal_rank_fusion(rankings: list[list[str]], offset: int = 60) -> list[str]:
    if offset < 0:
        raise ValueError("offset cannot be negative")
    combined: dict[str, float] = {}
    for ranking in rankings:
        if len(ranking) != len(set(ranking)):
            raise ValueError("duplicate document IDs within a ranking")
        for rank, doc_id in enumerate(ranking, start=1):
            combined[doc_id] = combined.get(doc_id, 0) + 1 / (offset + rank)
    return sorted(combined, key=lambda doc_id: (-combined[doc_id], doc_id))


class HybridRetriever:
    def __init__(self, documents: list[Document], dimensions: int = 6):
        self.documents = list(documents)
        self.bm25 = BM25(documents)
        self.latent = LatentSearch(documents, dimensions)

    def search(self, query: str, method: str = "hybrid", k: int = 5) -> list[str]:
        if k < 1 or k > len(self.documents):
            raise ValueError("k must lie between 1 and document count")
        lexical = order(self.bm25.scores(query), self.documents)
        low_rank = order(self.latent.scores(query), self.documents)
        if method == "bm25":
            ranked = lexical
        elif method == "latent":
            ranked = low_rank
        elif method == "hybrid":
            ranked = reciprocal_rank_fusion([lexical, low_rank])
        else:
            raise ValueError("method must be bm25, latent, or hybrid")
        return ranked[:k]

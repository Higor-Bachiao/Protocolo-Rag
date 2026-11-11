"""Recuperação: esparsa (TF-IDF/BM25-like), densa (embeddings) e híbrida, com reranking opcional.

- "esparsa": TF-IDF com sublinear tf, roda offline, sem GPU.
- "densa": embeddings multilíngues (padrão BAAI/bge-m3) via sentence-transformers.
- "hibrida": fusão por Reciprocal Rank Fusion (RRF) das duas listas.
- reranker: cross-encoder (padrão BAAI/bge-reranker-v2-m3) reordena o top-N.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chunking import Chunk


@dataclass
class Resultado:
    chunk: Chunk
    score: float


class RecuperadorEsparso:
    def __init__(self, chunks: list[Chunk]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.chunks = chunks
        self.vec = TfidfVectorizer(sublinear_tf=True, strip_accents="unicode", lowercase=True, ngram_range=(1, 2), min_df=1)
        self.matriz = self.vec.fit_transform([c.texto for c in chunks])

    def buscar(self, pergunta: str, k: int = 5) -> list[Resultado]:
        q = self.vec.transform([pergunta])
        scores = (self.matriz @ q.T).toarray().ravel()
        idx = np.argsort(-scores)[:k]
        return [Resultado(self.chunks[i], float(scores[i])) for i in idx]


class RecuperadorDenso:
    def __init__(self, chunks: list[Chunk], modelo: str = "BAAI/bge-m3", lote: int = 16):
        from sentence_transformers import SentenceTransformer

        self.chunks = chunks
        self.modelo = SentenceTransformer(modelo)
        self.emb = self.modelo.encode(
            [c.texto for c in chunks], batch_size=lote, normalize_embeddings=True, show_progress_bar=True
        )

    def buscar(self, pergunta: str, k: int = 5) -> list[Resultado]:
        q = self.modelo.encode([pergunta], normalize_embeddings=True)[0]
        scores = self.emb @ q
        idx = np.argsort(-scores)[:k]
        return [Resultado(self.chunks[i], float(scores[i])) for i in idx]


class RecuperadorHibrido:
    """Reciprocal Rank Fusion: score = soma 1/(60 + posição)."""

    def __init__(self, esparso: RecuperadorEsparso, denso: RecuperadorDenso, k_rrf: int = 60):
        self.esparso, self.denso, self.k_rrf = esparso, denso, k_rrf

    def buscar(self, pergunta: str, k: int = 5, profundidade: int = 30) -> list[Resultado]:
        acumulado: dict[str, float] = {}
        por_id: dict[str, Chunk] = {}
        for lista in (self.esparso.buscar(pergunta, profundidade), self.denso.buscar(pergunta, profundidade)):
            for pos, r in enumerate(lista):
                acumulado[r.chunk.chunk_id] = acumulado.get(r.chunk.chunk_id, 0.0) + 1.0 / (self.k_rrf + pos + 1)
                por_id[r.chunk.chunk_id] = r.chunk
        melhores = sorted(acumulado.items(), key=lambda x: -x[1])[:k]
        return [Resultado(por_id[cid], s) for cid, s in melhores]


class Reranker:
    def __init__(self, modelo: str = "BAAI/bge-reranker-v2-m3"):
        from sentence_transformers import CrossEncoder

        self.modelo = CrossEncoder(modelo)

    def reordenar(self, pergunta: str, resultados: list[Resultado], k: int) -> list[Resultado]:
        if not resultados:
            return resultados
        scores = self.modelo.predict([(pergunta, r.chunk.texto) for r in resultados])
        ordem = np.argsort(-np.asarray(scores))[:k]
        return [Resultado(resultados[i].chunk, float(scores[i])) for i in ordem]


def construir(chunks: list[Chunk], modo: str = "hibrida", modelo_emb: str = "BAAI/bge-m3"):
    if modo == "esparsa":
        return RecuperadorEsparso(chunks)
    if modo == "densa":
        return RecuperadorDenso(chunks, modelo_emb)
    if modo == "hibrida":
        return RecuperadorHibrido(RecuperadorEsparso(chunks), RecuperadorDenso(chunks, modelo_emb))
    raise ValueError(f"modo desconhecido: {modo}")

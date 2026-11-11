"""Montagem do RAG e do baseline (prompt simples, sem recuperação)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .llm import LLM
from .retriever import Reranker, Resultado

SISTEMA_RAG = (
    "Você responde perguntas sobre documentos públicos do Ministério da Saúde. "
    "Use SOMENTE os trechos fornecidos. Cite a fonte de cada afirmação no formato [n]. "
    "Se os trechos não contiverem a resposta, diga exatamente: "
    "'Não encontrei essa informação nos documentos.' Não use conhecimento próprio. "
    "Responda em português, em no máximo 5 frases."
)

SISTEMA_BASELINE = (
    "Você responde perguntas sobre normas e protocolos do SUS/Ministério da Saúde. "
    "Se não souber, diga que não sabe. Responda em português, em no máximo 5 frases."
)


@dataclass
class RespostaRAG:
    pergunta: str
    resposta: str
    trechos: list[Resultado] = field(default_factory=list)
    t_recuperacao: float = 0.0
    t_geracao: float = 0.0

    @property
    def t_total(self) -> float:
        return self.t_recuperacao + self.t_geracao


def formatar_contexto(trechos: list[Resultado]) -> str:
    return "\n".join(
        f"[{i}] ({r.chunk.titulo}, p. {r.chunk.pagina}) {r.chunk.texto}" for i, r in enumerate(trechos, 1)
    )


class PipelineRAG:
    def __init__(self, recuperador, llm: LLM, k: int = 4, reranker: Reranker | None = None, candidatos: int = 20):
        self.recuperador, self.llm, self.k = recuperador, llm, k
        self.reranker, self.candidatos = reranker, candidatos

    def responder(self, pergunta: str) -> RespostaRAG:
        t0 = time.perf_counter()
        if self.reranker:
            trechos = self.reranker.reordenar(pergunta, self.recuperador.buscar(pergunta, self.candidatos), self.k)
        else:
            trechos = self.recuperador.buscar(pergunta, self.k)
        t_rec = time.perf_counter() - t0

        usuario = f"Trechos:\n{formatar_contexto(trechos)}\n\nPergunta: {pergunta}"
        r = self.llm.gerar(SISTEMA_RAG, usuario)
        return RespostaRAG(pergunta, r.texto.strip(), trechos, t_rec, r.segundos)


class PipelineBaseline:
    """Prompt simples: o mesmo modelo, sem documentos."""

    def __init__(self, llm: LLM):
        self.llm = llm

    def responder(self, pergunta: str) -> RespostaRAG:
        r = self.llm.gerar(SISTEMA_BASELINE, pergunta)
        return RespostaRAG(pergunta, r.texto.strip(), [], 0.0, r.segundos)

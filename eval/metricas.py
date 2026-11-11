"""Métricas de avaliação.

- acerto_recuperacao: algum dos k trechos vem do documento-gabarito E contém a evidência esperada.
- fidelidade (estilo RAGAS, Es et al., 2024): o juiz quebra a resposta em afirmações e
  verifica quais são sustentadas pelos trechos recuperados. Fidelidade = sustentadas / total.
- correcao: o juiz compara a resposta com o gabarito -> 1 (correta), 0.5 (parcial), 0 (errada).
- abstencao: a resposta declarou que não encontrou a informação.
"""
from __future__ import annotations

import json
import re
import unicodedata

from rag.llm import LLM
from rag.pipeline import formatar_contexto, RespostaRAG

FRASE_ABSTENCAO = "não encontrei essa informação"


def normalizar(t: str) -> str:
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", t).strip()


def acerto_recuperacao(resp: RespostaRAG, doc_esperado: str | None, evidencia: str | None) -> float | None:
    """None quando a pergunta não tem resposta no corpus (não dá para errar a recuperação)."""
    if not doc_esperado:
        return None
    ev = normalizar(evidencia or "")
    for r in resp.trechos:
        if r.chunk.doc_id == doc_esperado and (not ev or ev in normalizar(r.chunk.texto)):
            return 1.0
    return 0.0


def abstencao(texto: str) -> bool:
    return normalizar(FRASE_ABSTENCAO) in normalizar(texto)


def _json_do_texto(texto: str):
    m = re.search(r"\{.*\}", texto, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


JUIZ_FIDELIDADE = (
    "Você é um avaliador rigoroso. Quebre a RESPOSTA em afirmações factuais atômicas. "
    "Para cada uma, decida se ela é sustentada EXPLICITAMENTE pelo CONTEXTO (não use conhecimento próprio). "
    'Devolva apenas JSON: {"afirmacoes": [{"texto": "...", "sustentada": true|false}]}. '
    'Se a resposta só diz que não encontrou a informação, devolva {"afirmacoes": []}.'
)


def fidelidade(juiz: LLM, resp: RespostaRAG) -> tuple[float | None, list]:
    if not resp.trechos or abstencao(resp.resposta):
        return None, []
    usuario = f"CONTEXTO:\n{formatar_contexto(resp.trechos)}\n\nRESPOSTA:\n{resp.resposta}"
    dados = _json_do_texto(juiz.gerar(JUIZ_FIDELIDADE, usuario, max_tokens=900).texto) or {}
    afirm = dados.get("afirmacoes", [])
    if not afirm:
        return None, []
    return sum(bool(a.get("sustentada")) for a in afirm) / len(afirm), afirm


JUIZ_CORRECAO = (
    "Compare a RESPOSTA com o GABARITO para a PERGUNTA. "
    "Nota 1 se a resposta contém a informação essencial do gabarito sem contradizê-lo; "
    "0.5 se está incompleta mas sem erro; 0 se está errada, contradiz o gabarito ou não responde. "
    "Se o gabarito diz que a informação NÃO está nos documentos, nota 1 apenas se a resposta se abstiver. "
    'Devolva apenas JSON: {"nota": 0|0.5|1, "motivo": "..."}'
)


def correcao(juiz: LLM, pergunta: str, resposta: str, gabarito: str) -> tuple[float | None, str]:
    usuario = f"PERGUNTA: {pergunta}\nGABARITO: {gabarito}\nRESPOSTA: {resposta}"
    dados = _json_do_texto(juiz.gerar(JUIZ_CORRECAO, usuario, max_tokens=300).texto) or {}
    try:
        return float(dados["nota"]), dados.get("motivo", "")
    except (KeyError, TypeError, ValueError):
        return None, "juiz não devolveu JSON válido"

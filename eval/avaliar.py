"""Avalia uma configuração do RAG (ou o baseline) nas perguntas do gabarito.

Exemplos:
  python -m eval.avaliar --modo hibrida --tamanho 300 --rerank
  python -m eval.avaliar --baseline
Saída em resultados/<nome>/: respostas.csv, tabela_perguntas.md, resumo.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from rag.chunking import dividir_corpus
from rag.ingest import carregar_corpus
from rag.llm import LLM
from rag.pipeline import PipelineBaseline, PipelineRAG
from rag.retriever import Reranker, construir
from eval import metricas

RAIZ = Path(__file__).resolve().parent.parent


def carregar_perguntas(caminho: Path) -> list[dict]:
    perguntas = json.loads(caminho.read_text(encoding="utf-8"))
    pendentes = [p["id"] for p in perguntas if "PREENCHER" in json.dumps(p, ensure_ascii=False)]
    if pendentes:
        raise SystemExit(f"Gabarito incompleto nas perguntas {pendentes}. Preencha eval/perguntas.json antes de avaliar.")
    return perguntas


def nome_config(a) -> str:
    if a.baseline:
        return "baseline_prompt_simples"
    return f"rag_{a.modo}_c{a.tamanho}_k{a.k}" + ("_rerank" if a.rerank else "")


def rodar(a, docs=None) -> dict:
    llm = LLM(temperatura=0.0)
    juiz = LLM(modelo=a.modelo_juiz) if a.modelo_juiz else llm
    perguntas = carregar_perguntas(Path(a.perguntas))
    docs = docs if docs is not None else carregar_corpus(a.corpus)
    if not docs:
        raise SystemExit(f"Nenhum PDF em {a.corpus}. Baixe os documentos listados em corpus/fontes.csv.")

    if a.baseline:
        pipe = PipelineBaseline(llm)
        n_chunks = 0
    else:
        chunks = dividir_corpus(docs, a.tamanho, a.sobreposicao)
        n_chunks = len(chunks)
        rec = construir(chunks, a.modo, a.modelo_emb)
        pipe = PipelineRAG(rec, llm, k=a.k, reranker=Reranker(a.modelo_rerank) if a.rerank else None)

    linhas = []
    for p in perguntas:
        r = pipe.responder(p["pergunta"])
        fid, afirm = metricas.fidelidade(juiz, r) if not a.baseline else (None, [])
        nota, motivo = metricas.correcao(juiz, p["pergunta"], r.resposta, p["gabarito"])
        top = r.trechos[0] if r.trechos else None
        linhas.append({
            "id": p["id"],
            "pergunta": p["pergunta"],
            "gabarito": p["gabarito"],
            "resposta": r.resposta,
            "trecho_recuperado": (top.chunk.texto[:500] if top else ""),
            "fonte_trecho": (f"{top.chunk.doc_id}, p. {top.chunk.pagina}" if top else ""),
            "docs_recuperados": ";".join(t.chunk.chunk_id for t in r.trechos),
            "acerto_recuperacao": None if a.baseline else metricas.acerto_recuperacao(r, p.get("doc_id"), p.get("evidencia")),
            "fidelidade": fid,
            "n_afirmacoes": len(afirm),
            "correcao": nota,
            "motivo_juiz": motivo,
            "absteve": metricas.abstencao(r.resposta),
            "t_recuperacao_s": round(r.t_recuperacao, 4),
            "t_geracao_s": round(r.t_geracao, 4),
            "t_total_s": round(r.t_total, 4),
        })

    df = pd.DataFrame(linhas)
    nome = nome_config(a)
    saida = RAIZ / "resultados" / nome
    saida.mkdir(parents=True, exist_ok=True)
    df.to_csv(saida / "respostas.csv", index=False, encoding="utf-8")

    def media(col):
        v = [x for x in df[col] if x is not None and pd.notna(x)]
        return round(statistics.mean(v), 3) if v else None

    resumo = {
        "config": nome,
        "provedor": llm.provedor,
        "modelo": llm.modelo,
        "modelo_juiz": juiz.modelo,
        "n_docs": len(docs),
        "n_chunks": n_chunks,
        "acerto_recuperacao": media("acerto_recuperacao"),
        "fidelidade": media("fidelidade"),
        "correcao": media("correcao"),
        "abstencoes": int(df["absteve"].sum()),
        "latencia_media_s": round(df["t_total_s"].mean(), 3),
        "latencia_p95_s": round(df["t_total_s"].quantile(0.95), 3),
        "latencia_recuperacao_media_s": round(df["t_recuperacao_s"].mean(), 4),
    }
    (saida / "resumo.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    escrever_tabela(df, saida / "tabela_perguntas.md")
    print(json.dumps(resumo, ensure_ascii=False, indent=2))
    return resumo


def escrever_tabela(df: pd.DataFrame, destino: Path) -> None:
    """Tabela exigida pelo enunciado: pergunta, resposta e trecho recuperado."""
    def curto(t, n=220):
        t = str(t).replace("|", "/").replace("\n", " ")
        return t if len(t) <= n else t[:n] + "…"

    linhas = ["| # | Pergunta | Resposta | Trecho recuperado (fonte) | Fidel. | Correção | Latência (s) |",
              "|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        fid = "—" if pd.isna(r["fidelidade"]) else f"{r['fidelidade']:.2f}"
        cor = "—" if pd.isna(r["correcao"]) else f"{r['correcao']:.1f}"
        linhas.append(
            f"| {r['id']} | {curto(r['pergunta'], 120)} | {curto(r['resposta'])} | "
            f"{curto(r['trecho_recuperado'], 160)} ({r['fonte_trecho']}) | {fid} | {cor} | {r['t_total_s']:.2f} |"
        )
    destino.write_text("\n".join(linhas) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(RAIZ / "corpus" / "pdfs"))
    ap.add_argument("--perguntas", default=str(RAIZ / "eval" / "perguntas.json"))
    ap.add_argument("--modo", default="hibrida", choices=["esparsa", "densa", "hibrida"])
    ap.add_argument("--tamanho", type=int, default=300, help="palavras por chunk")
    ap.add_argument("--sobreposicao", type=int, default=50)
    ap.add_argument("--k", type=int, default=4, help="trechos enviados ao LLM")
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--baseline", action="store_true", help="prompt simples, sem recuperação")
    ap.add_argument("--modelo-emb", default="BAAI/bge-m3")
    ap.add_argument("--modelo-rerank", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--modelo-juiz", default=None, help="modelo do juiz (padrão: o mesmo do LLM_MODEL)")
    return ap


if __name__ == "__main__":
    load_dotenv(RAIZ / ".env")
    rodar(parser().parse_args())

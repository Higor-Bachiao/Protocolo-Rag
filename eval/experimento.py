"""Roda o experimento completo e gera as figuras do relatório.

  python -m eval.experimento            # tudo (baseline + grade de configurações)
  python -m eval.experimento --rapido   # só baseline e a configuração principal

Configurações comparadas:
  - baseline (prompt simples, sem recuperação)
  - RAG esparso (TF-IDF), chunk 300
  - RAG híbrido (TF-IDF + bge-m3, RRF), chunks de 150, 300 e 600 palavras
  - RAG híbrido, chunk 300, com reranker
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv

from eval.avaliar import RAIZ, parser, rodar
from rag.ingest import carregar_corpus

GRADE = [
    {"baseline": True},
    {"modo": "esparsa", "tamanho": 300},
    {"modo": "hibrida", "tamanho": 150},
    {"modo": "hibrida", "tamanho": 300},
    {"modo": "hibrida", "tamanho": 600},
    {"modo": "hibrida", "tamanho": 300, "rerank": True},
]
RAPIDO = [GRADE[0], GRADE[-1]]


def rotulo(r: dict) -> str:
    c = r["config"]
    if c.startswith("baseline"):
        return "Prompt simples"
    partes = c.replace("rag_", "").split("_")
    return f"{partes[0]} {partes[1][1:]}p" + (" +rerank" if "rerank" in c else "")


def figuras(resumos: list[dict], destino: Path) -> None:
    df = pd.DataFrame(resumos)
    df["rotulo"] = df.apply(rotulo, axis=1)
    df.to_csv(destino / "comparacao.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    x = range(len(df))
    w = 0.38
    ax[0].bar([i - w / 2 for i in x], df["correcao"].fillna(0), w, label="Correção (vs. gabarito)")
    ax[0].bar([i + w / 2 for i in x], df["fidelidade"].fillna(0), w, label="Fidelidade (vs. trechos)")
    ax[0].set_xticks(list(x), df["rotulo"], rotation=30, ha="right")
    ax[0].set_ylim(0, 1.05)
    ax[0].set_title("Qualidade das respostas (média em 10 perguntas)")
    ax[0].legend(fontsize=8)

    ax[1].bar(x, df["latencia_media_s"], color="#7a8b99")
    ax[1].set_xticks(list(x), df["rotulo"], rotation=30, ha="right")
    ax[1].set_ylabel("segundos")
    ax[1].set_title("Latência média por pergunta")
    fig.tight_layout()
    fig.savefig(destino / "figura_comparacao.png", dpi=200)

    linhas = ["| Configuração | Acerto recup. | Fidelidade | Correção | Abstenções | Latência média (s) | p95 (s) |",
              "|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        f = lambda v: "—" if v is None or pd.isna(v) else f"{v:.2f}"
        linhas.append(f"| {r['rotulo']} | {f(r['acerto_recuperacao'])} | {f(r['fidelidade'])} | {f(r['correcao'])} | "
                      f"{r['abstencoes']} | {r['latencia_media_s']:.2f} | {r['latencia_p95_s']:.2f} |")
    (destino / "tabela_comparacao.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true")
    args = ap.parse_args()

    base = parser().parse_args([])
    docs = carregar_corpus(base.corpus)
    print(f"{len(docs)} documentos no corpus")
    if len(docs) < 20:
        print("[aviso] o enunciado pede ao menos 20 documentos públicos.")

    resumos = []
    for cfg in (RAPIDO if args.rapido else GRADE):
        a = parser().parse_args([])
        for k, v in cfg.items():
            setattr(a, k, v)
        resumos.append(rodar(a, docs))

    destino = RAIZ / "resultados"
    figuras(resumos, destino)
    (destino / "comparacao.json").write_text(json.dumps(resumos, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nFiguras e tabelas em {destino}")


if __name__ == "__main__":
    load_dotenv(RAIZ / ".env")
    main()

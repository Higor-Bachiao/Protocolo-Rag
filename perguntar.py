"""Uso interativo (demonstração na apresentação).

  python perguntar.py
  python perguntar.py "Quais vacinas compõem o calendário da criança aos 2 meses?"
"""
import sys
from pathlib import Path

from dotenv import load_dotenv

from rag.chunking import dividir_corpus
from rag.ingest import carregar_corpus
from rag.llm import LLM
from rag.pipeline import PipelineRAG
from rag.retriever import Reranker, construir

RAIZ = Path(__file__).resolve().parent


def main():
    load_dotenv(RAIZ / ".env")
    docs = carregar_corpus(RAIZ / "corpus" / "pdfs")
    chunks = dividir_corpus(docs, 300, 50)
    print(f"{len(docs)} documentos, {len(chunks)} chunks. Indexando…")
    pipe = PipelineRAG(construir(chunks, "hibrida"), LLM(), k=4, reranker=Reranker())

    perguntas = sys.argv[1:] or iter(lambda: input("\nPergunta (Enter para sair): ").strip(), "")
    for p in perguntas:
        r = pipe.responder(p)
        print(f"\n{r.resposta}\n")
        for i, t in enumerate(r.trechos, 1):
            print(f"  [{i}] {t.chunk.titulo}, p. {t.chunk.pagina} (score {t.score:.3f})")
        print(f"  recuperação {r.t_recuperacao:.2f}s · geração {r.t_geracao:.2f}s")


if __name__ == "__main__":
    main()

"""Ajuda a montar o gabarito: mostra os trechos do corpus que mais batem com uma busca.
Roda offline (TF-IDF), sem custo de API.

  python -m scripts.inspecionar "intervalo entre doses da vacina"
  python -m scripts.inspecionar --stats        # tamanho de cada documento
"""
import sys
from pathlib import Path

from rag.chunking import dividir_corpus
from rag.ingest import carregar_corpus
from rag.retriever import RecuperadorEsparso

RAIZ = Path(__file__).resolve().parent.parent

docs = carregar_corpus(RAIZ / "corpus" / "pdfs")
if "--stats" in sys.argv:
    for d in docs:
        print(f"{d.doc_id:60s} {len(d.paginas):4d} pág. {len(d.texto.split()):8d} palavras")
    print(f"\nTotal: {len(docs)} documentos")
    sys.exit()

consulta = " ".join(a for a in sys.argv[1:] if not a.startswith("--"))
rec = RecuperadorEsparso(dividir_corpus(docs, 300, 50))
for r in rec.buscar(consulta, 5):
    print(f"\n== {r.chunk.doc_id}, p. {r.chunk.pagina} (score {r.score:.3f})\n{r.chunk.texto[:700]}")

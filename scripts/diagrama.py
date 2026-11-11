"""Gera docs/arquitetura.png (figura da arquitetura usada no relatório)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(12, 4.6))
ax.set_xlim(0, 12); ax.set_ylim(0, 4.6); ax.axis("off")

def caixa(x, y, w, h, titulo, sub, cor):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12", fc=cor, ec="#33414d", lw=1.2))
    ax.text(x + w/2, y + h*0.64, titulo, ha="center", va="center", fontsize=10.5, weight="bold", color="#1d2730")
    ax.text(x + w/2, y + h*0.3, sub, ha="center", va="center", fontsize=8.3, color="#33414d")

def seta(x1, y1, x2, y2, txt=""):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", color="#33414d", lw=1.3))
    if txt: ax.text((x1+x2)/2, (y1+y2)/2 + 0.14, txt, ha="center", fontsize=7.8, color="#5b6b78")

ax.text(0.1, 4.35, "Indexação (offline)", fontsize=9, color="#5b6b78", style="italic")
caixa(0.1, 3.0, 2.0, 1.1, "Corpus", "≥20 PDFs públicos\nMin. da Saúde", "#e8eef3")
caixa(2.6, 3.0, 2.0, 1.1, "Extração", "pypdf + limpeza\npágina por trecho", "#e8eef3")
caixa(5.1, 3.0, 2.0, 1.1, "Chunking", "300 palavras\nsobreposição 50", "#e8eef3")
caixa(7.6, 3.4, 2.0, 0.75, "Índice léxico", "TF-IDF (1-2 gramas)", "#dfe9e1")
caixa(7.6, 2.5, 2.0, 0.75, "Índice vetorial", "bge-m3, normalizado", "#dfe9e1")
seta(2.1, 3.55, 2.6, 3.55); seta(4.6, 3.55, 5.1, 3.55)
seta(7.1, 3.7, 7.6, 3.75); seta(7.1, 3.3, 7.6, 2.9)

ax.text(0.1, 1.85, "Consulta (online)", fontsize=9, color="#5b6b78", style="italic")
caixa(0.1, 0.4, 2.0, 1.1, "Pergunta", "usuário", "#f4ece0")
caixa(2.6, 0.4, 2.2, 1.1, "Busca híbrida", "top-30 de cada índice\nfusão RRF", "#f4ece0")
caixa(5.3, 0.4, 2.0, 1.1, "Reranker", "cross-encoder\ntop-20 → k=4", "#f4ece0")
caixa(7.8, 0.4, 2.0, 1.1, "LLM", "temp. 0, cita [n]\nou se abstém", "#f4ece0")
caixa(10.3, 0.4, 1.6, 1.1, "Resposta", "+ trechos\n+ páginas", "#f4ece0")
seta(2.1, 0.95, 2.6, 0.95); seta(4.8, 0.95, 5.3, 0.95); seta(7.3, 0.95, 7.8, 0.95); seta(9.8, 0.95, 10.3, 0.95)
seta(8.6, 2.5, 3.9, 1.5, "consulta aos índices")

caixa(10.15, 2.5, 1.75, 1.6, "Avaliação", "10 perguntas\nfidelidade\ncorreção\nlatência", "#ece6f2")
seta(11.1, 1.5, 11.1, 2.5)
fig.tight_layout()
fig.savefig("docs/arquitetura.png", dpi=220)

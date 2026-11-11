"""Leitura dos PDFs/TXTs do corpus e limpeza básica do texto."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Documento:
    doc_id: str
    titulo: str
    texto: str
    paginas: list[int]  # deslocamento (em caracteres) do início de cada página


def _limpar(texto: str) -> str:
    texto = texto.replace("­", "")              # hífen suave
    texto = re.sub(r"-\n(\w)", r"\1", texto)          # palavra quebrada no fim da linha
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


def ler_pdf(caminho: Path) -> Documento:
    from pypdf import PdfReader

    leitor = PdfReader(str(caminho))
    partes, inicios, pos = [], [], 0
    for pagina in leitor.pages:
        t = _limpar(pagina.extract_text() or "")
        inicios.append(pos)
        partes.append(t)
        pos += len(t) + 2
    return Documento(caminho.stem, caminho.stem, "\n\n".join(partes), inicios)


def ler_txt(caminho: Path) -> Documento:
    t = _limpar(caminho.read_text(encoding="utf-8"))
    return Documento(caminho.stem, caminho.stem, t, [0])


def carregar_corpus(pasta: str | Path) -> list[Documento]:
    pasta = Path(pasta)
    if not pasta.is_dir():
        return []
    docs = []
    for arq in sorted(pasta.iterdir()):
        if arq.suffix.lower() == ".pdf":
            docs.append(ler_pdf(arq))
        elif arq.suffix.lower() in {".txt", ".md"}:
            docs.append(ler_txt(arq))
    vazios = [d.doc_id for d in docs if len(d.texto) < 200]
    if vazios:
        print(f"[aviso] pouco texto extraído (PDF escaneado?): {vazios}")
    return [d for d in docs if d.texto]


def pagina_de(doc: Documento, offset: int) -> int:
    """Número da página (1-based) que contém o caractere `offset`."""
    n = 1
    for i, ini in enumerate(doc.paginas):
        if offset >= ini:
            n = i + 1
    return n


def salvar_jsonl(docs: list[Documento], destino: Path) -> None:
    with open(destino, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")

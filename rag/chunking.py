"""Divisão dos documentos em chunks.

Tamanho medido em palavras (aprox. 0,75 palavra por token em português),
com sobreposição para não cortar uma informação ao meio. O corte procura
o fim de frase mais próximo para não quebrar sentenças.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .ingest import Documento, pagina_de


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    titulo: str
    pagina: int
    texto: str


_FIM_FRASE = re.compile(r"(?<=[.;:!?])\s+")


def dividir(doc: Documento, tamanho: int = 300, sobreposicao: int = 50) -> list[Chunk]:
    if sobreposicao >= tamanho:
        raise ValueError("sobreposicao deve ser menor que tamanho")

    # tokens com a posição em caracteres para recuperar a página de origem
    tokens = [(m.group(), m.start()) for m in re.finditer(r"\S+", doc.texto)]
    chunks, ini, n = [], 0, 0
    while ini < len(tokens):
        fim = min(ini + tamanho, len(tokens))
        # recua até um fim de frase, se houver um nos últimos 20% do chunk
        if fim < len(tokens):
            for j in range(fim, ini + int(tamanho * 0.8), -1):
                if tokens[j - 1][0][-1] in ".;:!?":
                    fim = j
                    break
        texto = " ".join(t for t, _ in tokens[ini:fim])
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}#{n}",
            doc_id=doc.doc_id,
            titulo=doc.titulo,
            pagina=pagina_de(doc, tokens[ini][1]),
            texto=texto,
        ))
        n += 1
        if fim == len(tokens):
            break
        ini = max(fim - sobreposicao, ini + 1)
    return chunks


def dividir_corpus(docs: list[Documento], tamanho: int = 300, sobreposicao: int = 50) -> list[Chunk]:
    return [c for d in docs for c in dividir(d, tamanho, sobreposicao)]

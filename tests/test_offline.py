"""Testes que rodam sem internet e sem chave de API: pytest -q"""
from rag.chunking import dividir
from rag.ingest import Documento
from rag.llm import LLM
from rag.pipeline import PipelineRAG
from rag.retriever import RecuperadorEsparso
from eval import metricas

TEXTO = (
    "A vacina BCG deve ser aplicada ao nascer, em dose única. "
    "A hepatite B tem a primeira dose nas primeiras 24 horas de vida. " * 3
    + "O protocolo de hipertensão recomenda medir a pressão arterial em duas consultas. " * 40
)
DOC = Documento("doc_teste", "Documento de teste", TEXTO, [0, len(TEXTO) // 2])


def test_chunks_cobrem_texto_e_respeitam_tamanho():
    chunks = dividir(DOC, tamanho=40, sobreposicao=10)
    assert len(chunks) > 3
    assert all(len(c.texto.split()) <= 40 for c in chunks)
    assert chunks[-1].texto.split()[-1] == TEXTO.split()[-1]
    assert chunks[-1].pagina == 2


def test_sobreposicao_invalida():
    try:
        dividir(DOC, tamanho=10, sobreposicao=10)
        assert False
    except ValueError:
        pass


def test_recuperacao_esparsa_acha_documento_certo():
    rec = RecuperadorEsparso(dividir(DOC, 40, 10))
    top = rec.buscar("quando aplicar a vacina BCG", k=1)[0]
    assert "BCG" in top.chunk.texto


def test_pipeline_ponta_a_ponta_sem_api():
    rec = RecuperadorEsparso(dividir(DOC, 40, 10))
    r = PipelineRAG(rec, LLM(provedor="eco"), k=2).responder("vacina BCG")
    assert r.trechos and r.resposta
    assert metricas.acerto_recuperacao(r, "doc_teste", "dose única") == 1.0
    assert metricas.acerto_recuperacao(r, "outro_doc", None) == 0.0
    assert metricas.acerto_recuperacao(r, None, None) is None


def test_abstencao_ignora_acentos():
    assert metricas.abstencao("Nao encontrei essa informacao nos documentos.")
    assert not metricas.abstencao("A dose é única.")

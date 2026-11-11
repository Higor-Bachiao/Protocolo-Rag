# RAG sobre documentos públicos do SUS

Trabalho semestral de Tópicos Avançados em Computação (Tema 01 · RAG, modelos de linguagem e agentes) — Unifeg.
Autores: Higor Bachião, Olímpio João e Gabriel Vasconcelos.

**Pergunta de pesquisa:** em perguntas sobre protocolos públicos do Ministério da Saúde, quanto a recuperação de trechos (RAG) melhora a correção e a fidelidade das respostas de um LLM em relação ao prompt simples, e a que custo de latência?

## O que tem aqui

```
rag/ingest.py      leitura dos PDFs e limpeza do texto (guarda a página de cada trecho)
rag/chunking.py    divisão em chunks por palavras, com sobreposição e corte em fim de frase
rag/retriever.py   busca esparsa (TF-IDF), densa (bge-m3), híbrida (RRF) e reranker (cross-encoder)
rag/llm.py         cliente OpenAI / Gemini / Anthropic, temperatura 0, cache em disco
rag/pipeline.py    RAG com citação [n] e abstenção; baseline de prompt simples
eval/perguntas.json  10 perguntas com gabarito, documento e evidência
eval/metricas.py   acerto de recuperação, fidelidade (estilo RAGAS), correção, abstenção
eval/avaliar.py    avalia uma configuração
eval/experimento.py  roda a grade completa e gera tabela + figura
perguntar.py       modo interativo para a demonstração
scripts/inspecionar.py  busca offline no corpus para montar o gabarito
corpus/fontes.csv  lista dos documentos usados, com URL e data de acesso
```

## Reproduzir

Requer Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                 # preencher LLM_PROVIDER, LLM_MODEL e a chave
pytest -q                            # testes offline, sem API
```

1. Baixe os PDFs listados em `corpus/fontes.csv` para `corpus/pdfs/` com o nome da coluna `arquivo` (ex.: `corpus/pdfs/pcdt_2018_completo.pdf`). Mínimo de 20 documentos.
2. Confira a extração: `python -m scripts.inspecionar --stats` (documento com poucas palavras = PDF escaneado; troque por outro).
3. Preencha `eval/perguntas.json`. Use `python -m scripts.inspecionar "termo"` para achar o trecho e a página. O campo `evidencia` precisa ser um pedaço **literal** do documento.
4. Rode o experimento:

```bash
python -m eval.experimento
```

Comando exato usado nos resultados do relatório: `python -m eval.experimento` (preencher aqui o provedor e o modelo usados, ex.: `LLM_PROVIDER=... LLM_MODEL=...`).

Saídas em `resultados/`: `tabela_comparacao.md`, `figura_comparacao.png`, e uma pasta por configuração com `respostas.csv`, `tabela_perguntas.md` (pergunta, resposta e trecho recuperado) e `resumo.json`.

Na primeira execução os modelos `BAAI/bge-m3` (~2,3 GB) e `BAAI/bge-reranker-v2-m3` são baixados do Hugging Face. Em CPU a indexação leva alguns minutos; para testar rápido use `python -m eval.experimento --rapido`.

## Decisões técnicas

| Decisão | Escolha | Motivo |
|---|---|---|
| Chunk | 300 palavras, sobreposição 50 | meio-termo entre contexto e precisão; testado contra 150 e 600 |
| Embedding | BAAI/bge-m3 | multilíngue (português), documentos longos |
| Busca | híbrida TF-IDF + densa, fusão RRF | siglas e nomes de medicamento casam melhor no léxico; sinônimos no denso |
| Reranker | bge-reranker-v2-m3 sobre top-20 | melhora a ordem antes de cortar em k=4 |
| k | 4 trechos | limita contexto (efeito "lost in the middle") |
| Geração | temperatura 0, citação [n], frase fixa de abstenção | reprodutibilidade e verificabilidade |
| Juiz | mesmo LLM (configurável com `--modelo-juiz`) | limitação declarada no relatório |

## O que ficou de fora de propósito

Fine-tuning (só discutido, não treinado), OCR de PDFs escaneados, tabelas extraídas de forma estruturada, interface web e agentes com ferramentas.

## Dados

Somente documentos públicos do Ministério da Saúde. Nenhum dado pessoal. O sistema é um protótipo acadêmico e **não** deve ser usado para decisão clínica.
# Protocolo-Rag

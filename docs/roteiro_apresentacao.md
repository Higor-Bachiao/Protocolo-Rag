# Roteiro da apresentação (15 min, máximo 10 slides)

| # | Slide | Conteúdo | Quem | Tempo |
|---|---|---|---|---|
| 1 | Título | Tema, nomes, Unifeg | PREENCHER | 0:30 |
| 2 | Problema | LLM inventa dose/critério; PCDT longos e atualizados. Pergunta de pesquisa em uma frase | PREENCHER | 1:30 |
| 3 | Como o RAG funciona | Figura de arquitetura (docs/arquitetura.png). É a figura que conduz a fala | PREENCHER | 2:00 |
| 4 | Escolhas técnicas | Chunk 300, híbrida + RRF, reranker, k=4, abstenção. Uma linha de "por quê" para cada | PREENCHER | 1:30 |
| 5 | Método | Corpus, 10 perguntas (inclui a sem resposta), 6 configurações, 4 métricas | PREENCHER | 1:30 |
| 6 | Resultado principal | figura_comparacao.png + a frase com números | PREENCHER | 2:00 |
| 7 | Um acerto e um erro | Uma pergunta que o RAG acertou e o prompt simples errou; um erro do RAG com o trecho recuperado | PREENCHER | 2:00 |
| 8 | Demonstração | `python perguntar.py "..."` ao vivo (ter um vídeo de reserva) | PREENCHER | 1:30 |
| 9 | Limites | n=10, juiz = gerador, contaminação do baseline, não é uso clínico | PREENCHER | 1:30 |
| 10 | Conclusão | O que mediram, o que não dá para concluir, próximos passos | PREENCHER | 1:00 |

Regra para a fala: ao citar um número, dizer se foi **medido** por vocês ou **lido** em artigo. A rubrica cobra isso.

## Perguntas prováveis da arguição

**Por que chunk de 300 palavras?**
Meio-termo: cabe uma seção curta de PCDT sem separar a dose da condição, e 4 chunks cabem com folga no contexto. Não foi chute: testamos 150 e 600 e o resultado foi PREENCHER.

**Por que essas métricas?**
Correção sozinha não diz onde o erro nasceu. Acerto de recuperação isola a busca; fidelidade isola a geração (resposta fiel ao trecho errado = problema da busca; trecho certo e resposta infiel = problema do modelo). Latência é o custo.

**O que muda se o corpus crescer?**
Mais chunks parecidos competem no top-k, a precisão da busca léxica cai e o reranker passa a pesar mais. A latência de recuperação cresce linearmente na busca exaustiva; com centenas de milhares de chunks seria preciso índice aproximado (HNSW, FAISS). A geração não muda porque k continua 4. A medida entre chunk 150 (mais chunks) e 600 (menos) mostrou PREENCHER.

**O juiz é o mesmo modelo. Isso não invalida?**
Não invalida, mas enviesa. Por isso revisamos à mão as notas da configuração principal e registramos PREENCHER discordâncias. Com mais tempo, usaríamos um juiz de outro provedor (o código aceita `--modelo-juiz`).

**Por que não fine-tuning?**
Ovadia et al. (2024) mostram RAG superior para injetar conhecimento; fine-tuning não dá a fonte e exige retreino a cada atualização de PCDT. Não medimos, então não afirmamos que seria pior aqui.

**10 perguntas bastam?**
Não para afirmar significância: uma pergunta move a média em 0,1. Bastam para comparar comportamentos e classificar tipos de erro, que é o que fizemos.

**E se o modelo já sabia a resposta?**
É a contaminação do baseline: PCDT são públicos e podem estar no treino. Por isso a pergunta 10, fora do corpus, e a comparação de abstenção são importantes.

**Por que busca híbrida e não só embeddings?**
PCDT têm siglas, CID e nomes de medicamento, onde a busca léxica é forte; a densa pega sinônimos. A pergunta 7 (sinônimos) e a comparação léxica × híbrida mostram PREENCHER.

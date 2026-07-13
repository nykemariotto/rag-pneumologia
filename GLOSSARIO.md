# Glossário: em português simples

> Termos de computação (e em inglês) que aparecem no projeto, explicados sem jargão.
> A ideia central: o assistente faz uma **"prova com consulta"**: em vez de responder
> "de cabeça", ele consulta uma estante de textos confiáveis antes de responder.

## Os 3 conceitos centrais
| Termo | Em português simples |
|---|---|
| **LLM** | A "inteligência" que escreve as respostas, no nosso caso o **Qwen2.5** (modelo aberto, roda localmente, sem chave). É o "aluno" que responde. |
| **KB** / *knowledge base* / base de conhecimento | A **estante de materiais confiáveis** que o assistente consulta: os nossos **33 textos** sobre DPOC, fibrose e PCM. |
| **RAG** (*Retrieval-Augmented Generation*) | A técnica de **deixar o LLM consultar a estante** antes de responder. É o coração do trabalho. |

## As fontes (de onde vêm os textos)
| Termo | O que é |
|---|---|
| **StatPearls** | Site **gratuito** de artigos médicos revisados, uma enciclopédia clínica aberta. |
| **WHO** | É a **OMS** (Organização Mundial da Saúde), em inglês. |
| **PMC** (*PubMed Central*) | Arquivo **gratuito** de artigos médicos completos (governo dos EUA). |
| **GOLD / NICE** | **Diretrizes clínicas** famosas (GOLD = diretriz mundial de DPOC; NICE = do Reino Unido). |
| **"âncora" / "flagship" / carro-chefe** | Modo de dizer "**o documento principal**, mais importante". Nada técnico. |
| **licença / CC BY…** | A "regra de uso" de cada texto: se podemos **copiar/compartilhar** ele sem problema de direitos autorais. |

## As engrenagens (como a consulta funciona)
| Termo | Em português simples |
|---|---|
| **chunk / "pedaço"** | A gente **corta os textos em pedaços** (parágrafos); não dá pra entregar a estante inteira de uma vez. |
| **embedding** | Virar cada pedaço num **"código de significado"** (lista de números) pra achar pedaços **pelo sentido**, não só pela palavra. (Como uma rede neural vira uma imagem em vetor de características.) |
| **FAISS** / banco vetorial | A **"gaveta organizada"** que guarda esses códigos e acha os mais parecidos rapidinho. |
| **BM25** | Busca **por palavra-chave** (a busca "tradicional", tipo Ctrl+F). |
| **recuperação** / *retrieval* | **Pescar os pedaços mais relevantes** pra responder a pergunta. |
| **reranker** | Um **segundo leitor mais criterioso** que **reordena** os pedaços, pondo os melhores no topo. |
| **baseline** | A versão de **comparação**: responde **sem consultar** a estante (só de cabeça). |
| **LLM-as-judge** | Usar uma IA pra ajudar a **dar nota** nas respostas (além de você conferir uma amostra). |

## As 3 versões que o trabalho compara
1. **Baseline**: só o LLM (Qwen), sem estante (responde de cabeça).
2. **RAG convencional**: o LLM (Qwen) **com busca** na estante.
3. **Proposta**: busca **+ reordenação** (reranker). É a nossa contribuição "avançada".

# Assistente de Pneumologia com RAG
### Trabalho Final · AM1PIDPG-93 · Aprendizado de Máquina Supervisionado (UNESP)

Assistente de IA de domínio específico para **doenças pulmonares crônicas e difusas**,
construído com **Retrieval-Augmented Generation (RAG)**.

## Motivação
O domínio deste trabalho nasceu de uma colaboração clínica real. No **Laboratório de Física Médica
(IBB-UNESP, Botucatu)** atuei na **análise quantitativa das tomografias de tórax** (segmentação
pulmonar por *deep learning*, densitometria/quantificação de enfisema (%LAA) e coregistro longitudinal)
do doutorado da **Dra. Erika Mayumi Watanabe** (Programa de Pós-Graduação em Doenças Tropicais,
**Faculdade de Medicina de Botucatu, FMB/UNESP**; imagens do **HC-FMB**; orientação do **Prof. Ricardo
de Souza Cavalcante**, coorientação de Sérgio Ribeiro Marrone), **defendido em 09/06/2026**, sobre as
**sequelas pulmonares da paracoccidioidomicose (PCM)**, micose endêmica na região de Botucatu.
A coorte comparava **DPOC, PCM ativa e PCM residual**: as três doenças que se sobrepõem no pulmão e são
diagnóstico diferencial entre si. Lidando de perto com elas, ficou claro (a) o quanto exigem **consulta
constante a fontes confiáveis** e (b) o quanto a **PCM, por ser regional, é mal coberta pelas IAs
generalistas**. Daí a escolha do domínio deste assistente (**DPOC + fibrose pulmonar + PCM**), com
respostas sempre **ancoradas em fontes citáveis**.

## Domínio
Pneumologia: três doenças que compartilham apresentação (dispneia/tosse crônica + alteração
de imagem/prova de função) e são diagnóstico diferencial entre si:
- **DPOC** (+ enfisema)
- **Fibrose pulmonar** (FPI / doença pulmonar intersticial)
- **Paracoccidioidomicose (PCM)**: micose endêmica; a forma crônica causa fibrose/enfisema

Idioma do assistente e do benchmark: **PT-BR**. Knowledge base: bilíngue (PT + EN).

## Arquitetura: 3 configurações comparadas
| # | Configuração | Descrição |
|---|---|---|
| 1 | **Baseline** | LLM sem recuperação (prompt direto) |
| 2 | **RAG convencional** | embeddings densos → FAISS → top-k → LLM |
| 3 | **Proposta** | busca densa amplia o pool de candidatos → **reranker cross-encoder** reordena por relevância → top-k → LLM |

> A técnica avançada da proposta é o **reranker neural** (cross-encoder), que relê os candidatos e
> põe o trecho-chave no topo, recuperação mais precisa que a similaridade de embeddings sozinha.
> *(BM25 e busca híbrida (RRF) também foram implementados e explorados, ver `hybrid_search` em
> `rag_core.py`, mas injetavam trechos fora do tema que confundiam o modelo pequeno; a densa +
> reranker ficou mais limpa.)*

O modelo aberto **`Qwen2.5-3B-Instruct`** (Apache-2.0) roda na GPU (T4 do Colab ou GPU local) e faz
tanto a geração quanto a IA-juíza.

**Stack:** `transformers` (Qwen2.5-3B) · `sentence-transformers`
(embeddings `multilingual-e5`) · FAISS · `rank-bm25` · `bge-reranker-v2-m3` (reranker).

**Avaliação:** retrieval (hit-rate, MRR) · geração (LLM-as-judge + validação humana de subconjunto)
· custo (latência real, tokens). Comparação Baseline × RAG × Proposta. Geração **determinística**
(*greedy*, `do_sample=False`) → reprodutível.

## Principais resultados
- **Consultar fontes ajuda:** a qualidade sobe de 0,47 (baseline) para 0,57 (RAG); só as versões com RAG
  citam a fonte (0% → 33%) e são **fiéis** aos trechos (fidelidade 0,00 vs 0,90 e 0,93).
- **A proposta vence na recuperação** (hit-rate 37% vs 30%), mas no modelo de 3B esse ganho não chega à
  geração (proposta ≈ RAG na qualidade).
- **Ablação de escala:** com a mesma recuperação, num modelo de 7B o ganho **aparece**: a proposta supera
  o RAG (qualidade 0,81 vs 0,73). O reranking se traduz em respostas melhores quando o modelo é grande o
  suficiente para aproveitar o contexto.

## Estrutura de pastas
```
rag-pneumologia/
├── README.md
├── rag_core.py               # nucleo: 3 configs, busca, reranking, geracao
├── requirements.txt
├── FONTES.md                 # referencias + licencas
├── kb_sources.json           # manifesto curado de fontes (o "dataset")
├── notebook/                 # Assistente_Pneumologia_RAG.ipynb (roda de ponta a ponta)
├── scripts/                  # indexar.py, avaliar.py, resumo_resultados.py
├── knowledge_base/           # 33 documentos (txt), por doenca: dpoc/ fibrose/ pcm/
├── indices/                  # FAISS + BM25 + chunks (pre-construido)
├── benchmark/                # perguntas.json (54, com gabarito e fonte-ouro)
├── _resultados/              # resultados pre-computados (3B, 7B) + figuras
└── figuras/                  # figura metodologica
```

## Regras de licença da KB
Só entram na pasta documentos **redistribuíveis** (domínio público, CC BY/BY-NC/BY-NC-ND/BY-SA,
CC BY-NC-SA). Cada arquivo guarda fonte + licença. Documentos free-to-read porém **não
redistribuíveis** (GOLD, NICE, diretrizes IPF 2018/2015) entram só como **referência (link)**,
nunca como arquivo.

## Como rodar

### Google Colab (recomendado)
Abra o notebook direto no Colab por este link:

https://colab.research.google.com/github/nykemariotto/rag-pneumologia/blob/main/notebook/Assistente_Pneumologia_RAG.ipynb

Depois: **Runtime → Change runtime type → T4 GPU** e **Runtime → Run all**. A 1ª célula clona o
repositório sozinha (código + base + índice + benchmark + resultados); o modelo aberto (~6 GB) baixa uma vez.

### JupyterLab local
```bash
git clone https://github.com/nykemariotto/rag-pneumologia.git
cd rag-pneumologia
python -m venv .venv
.venv\Scripts\activate            # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt jupyterlab
# torch com CUDA (local), se ainda nao tiver:
#   pip install torch --index-url https://download.pytorch.org/whl/cu126
jupyter lab notebook/Assistente_Pneumologia_RAG.ipynb
```
A 1ª célula detecta que já está no repositório e usa os arquivos locais (não re-clona).

### Ablação de tamanho de modelo (opcional, 7B em 4-bit)
```bash
pip install bitsandbytes
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct LLM_4BIT=1 MODEL_TAG=7b python scripts/avaliar.py all
```

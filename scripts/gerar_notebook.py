"""
Gera o notebook (.ipynb) do trabalho, montando as celulas programaticamente.
Saida: notebook/Assistente_Pneumologia_RAG.ipynb
Roda em Colab (GPU T4 gratuita) OU em JupyterLab local -> a 1a celula clona o repo do GitHub
(ou usa os arquivos locais). Geracao 100% LOCAL (modelo aberto Qwen2.5), SEM chave de API.
Todo o codigo usado (rag_core, indexar, avaliar, resumo) e embutido no notebook (%%writefile),
para que o professor entenda TUDO so lendo o notebook, sem caçar arquivos.
"""
import os, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "notebook")
os.makedirs(OUT, exist_ok=True)
# codigos-fonte embutidos no notebook (o que voce le == o que roda)
RC_SRC = open(os.path.join(ROOT, "rag_core.py"), encoding="utf-8").read()
IDX_SRC = open(os.path.join(ROOT, "scripts", "indexar.py"), encoding="utf-8").read()
AVAL_SRC = open(os.path.join(ROOT, "scripts", "avaliar.py"), encoding="utf-8").read()
RESUMO_SRC = open(os.path.join(ROOT, "scripts", "resumo_resultados.py"), encoding="utf-8").read()

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}
def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}

cells = []

cells.append(md("""
# Assistente de Pneumologia com RAG
**Trabalho Final · Disciplina de Aprendizado de Máquina Supervisionado (UNESP)**

Autor: Nycolas Mariotto · Doutorando em Ciências Biomoleculares e Farmacológicas (UNESP/Botucatu)

Um assistente que responde perguntas clínicas sobre DPOC, fibrose pulmonar e
paracoccidioidomicose (PCM) baseado em fontes confiáveis e citáveis, usando
*Retrieval-Augmented Generation* (RAG).

**Objetivo:** medir se recuperar fontes (e reordená-las com *reranking*) torna as respostas mais
corretas e rastreáveis do que o LLM respondendo sozinho.

> **Motivação:** a ideia nasceu da minha participação no doutorado da Dra. Erika Mayumi Watanabe
> (FMB/HC-FMB, orientação do Prof. Ricardo de Souza Cavalcante, defendido em 09/06/2026) sobre as
> sequelas pulmonares da PCM, no qual fiz a análise quantitativa das tomografias. No pulmão, DPOC,
> enfisema, fibrose e PCM deixam marcas parecidas e se confundem; este assistente cobre DPOC, fibrose e PCM.

Comparei **3 configurações**:
1. **Baseline**: LLM sem recuperação (geração direta, paramétrica).
2. **RAG convencional**: recuperação densa (*embeddings* + FAISS) → LLM.
3. **Proposta**: recuperação densa + reranking por cross-encoder que reescora os candidatos e leva o
   trecho mais pertinente ao topo do contexto → LLM.

> Roda 100% localmente com um modelo aberto (`Qwen2.5-3B-Instruct`, Apache-2.0), na GPU gratuita
> do Colab (T4) ou numa GPU local. O modelo é configurável: além do 3B padrão, rodei também o 7B
> (ablação de tamanho de modelo, §5b). Todo o código está embutido neste notebook, nada fica escondido.
>
> Base de conhecimento: 33 documentos abertos (~670 páginas), PT/EN, recortados em 2.374 *chunks*.
> Avaliação em 54 perguntas PT-BR com gabarito rastreável à fonte.
"""))

cells.append(md("""
## Como executar
Funciona em Google Colab (grátis) ou JupyterLab local; a 1ª célula cuida de tudo.
1. (Colab) **Runtime → Change runtime type → T4 GPU**. A geração usa GPU; em CPU fica muito lento.
2. **Run all** (ou célula a célula).

A 1ª célula clona o projeto do GitHub (código + base + índice + benchmark + resultados), ou usa os
arquivos locais, se você já estiver dentro do repositório. O modelo aberto (~6 GB) baixa uma única vez
na primeira geração; o índice de busca já vem pronto (não re-processa nada).
"""))

cells.append(code('''
# 1) Carrega o projeto (codigo + base + indice + benchmark + resultados).
#    PORTATIL: se ja estiver dentro do repo (JupyterLab local, mesmo abrindo de notebook/),
#    usa os arquivos LOCAIS; senao (ex.: Colab novo), faz git clone. Idempotente.
import os, subprocess
REPO_URL = "https://github.com/nykemariotto/rag-pneumologia.git"
REPO_DIR = "rag-pneumologia"
def _tem_projeto(d="."):
    return os.path.exists(os.path.join(d, "rag_core.py")) and os.path.isdir(os.path.join(d, "indices"))
if _tem_projeto("."):
    pass                              # ja na raiz do projeto
elif _tem_projeto(".."):
    os.chdir("..")                    # aberto de dentro de notebook/ (JupyterLab local)
else:
    if not os.path.isdir(REPO_DIR):
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL], check=True)
    os.chdir(REPO_DIR)                # Colab / ambiente novo
assert _tem_projeto("."), "Projeto nao encontrado (esperado rag_core.py e a pasta indices/)."
print("Projeto carregado em:", os.getcwd())
'''))

cells.append(code("""
# 2) Instala as dependências (versões fixadas para reprodutibilidade) e confirma o ambiente.
#    O torch com GPU já vem pré-instalado no Colab; aqui instalamos só o restante.
!pip install -q -r requirements.txt
import torch
from importlib.metadata import version
libs = {p: version(p) for p in ["transformers", "sentence-transformers", "faiss-cpu", "accelerate", "rank-bm25"]}
print("Bibliotecas instaladas:", libs)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else "(NENHUMA — ative a T4 em Runtime > Change runtime type e rode tudo de novo)")
"""))

cells.append(md("""
## 1. Base de conhecimento (o *corpus*)
33 documentos abertos e redistribuíveis (~670 páginas), curados por confiabilidade e licença:
StatPearls, CDC/MedlinePlus/NHLBI (domínio público), OMS, diretrizes da SBPT (J Bras Pneumol),
Consenso Brasileiro de PCM, diretriz ATS/ERS de fibrose e revisões *open-access* (PMC/MDPI). Corpus
bilíngue (PT/EN); cada `.txt` retém atribuição e licença (ver `FONTES.md`). Diretrizes com *copyright*
(GOLD, NICE, ATS 2015/2018) entram só como referência citável, não no corpus.
"""))

cells.append(code("""
import json
from collections import Counter
meta = json.load(open("indices/meta.json"))
chunks = [json.loads(l) for l in open("indices/chunks.jsonl", encoding="utf-8")]
print(f"Documentos: {meta['n_docs']}  |  Trechos (chunks): {meta['n_chunks']}  |  modelo de embeddings: {meta['emb_model']}")
NOMES = {"dpoc": "DPOC", "fibrose": "Fibrose", "pcm": "PCM"}
cont = Counter(c["disease"] for c in chunks)
print("Trechos por doença:", {NOMES.get(k, k): v for k, v in cont.items()})
"""))

cells.append(md("""
### Como o índice é construído (`scripts/indexar.py`)
O código abaixo lê os `.txt`, corta em *chunks*, gera os *embeddings* e monta os índices FAISS (denso) e
BM25 (palavra-chave). Ele já foi rodado, o índice pronto vem no repositório, então o notebook não
re-processa (é só para o código ficar visível). A célula grava o arquivo, mas não o executa aqui.
"""))

cells.append(code("%%writefile scripts/indexar.py" + chr(10) + IDX_SRC))

cells.append(md("""
## 2. Arquitetura
Núcleo em `rag_core.py`; o índice é pré-construído por `scripts/indexar.py` e reusado aqui. O pipeline é
determinístico de ponta a ponta: a recuperação exata e a decodificação *greedy* (`do_sample=False`)
tornam os resultados reprodutíveis.
- **Indexação semântica:** os trechos (~1100 caracteres, chunking por parágrafo com sobreposição) são
  codificados por `intfloat/multilingual-e5-base` (prefixos `passage:` e `query:`), normalizados e indexados
  em FAISS `IndexFlatIP` (produto interno em vetores normalizados ≡ similaridade de cosseno).
- **Recuperação (as 3 configurações):** compartilham o índice e diferem só na seleção dos trechos:
  (1) **baseline**, sem recuperação; (2) **RAG**, top-k denso; (3) **proposta**, top-k denso ampliado e
  reordenado por um reranker cross-encoder (`BAAI/bge-reranker-v2-m3`), que reescora cada par
  (consulta, trecho) e promove o mais pertinente ao topo do contexto.
- **Decisão de projeto:** uma fusão híbrida densa+BM25 via *Reciprocal Rank Fusion* (`hybrid_search`) foi
  implementada e descartada, pois introduzia trechos fora de tópico que degradavam a geração no modelo de 3B;
  densa e reranker preservaram a precisão sem o ruído.
- **Geração:** `Qwen2.5-3B-Instruct` (Apache-2.0) servido localmente em fp16 na GPU; a mesma instância atua
  como LLM-as-judge na avaliação.
- **O que cada modelo aprendeu (*transfer learning*):** o pipeline compõe modelos pré-treinados, sem
  treino próprio. O *embedder* (`e5`) é um bi-encoder que aprendeu, por aprendizado contrastivo, a
  aproximar textos de sentido próximo; o reranker é um cross-encoder, um classificador de relevância
  que aprendeu, de pares (consulta, trecho) rotulados, a pontuar quão pertinente é um trecho; o LLM aprendeu
  modelagem de linguagem. Bi-encoder é rápido (busca); cross-encoder é preciso (reordenação).
"""))

cells.append(code('''
# Figura metodologica: as 3 configuracoes lado a lado (diferem so na SELECAO dos trechos que vao ao LLM).
from IPython.display import Image
Image(filename="figuras/pipeline_rag.png")
'''))

cells.append(md("""
### O núcleo (`rag_core.py`): implementação completa
A célula abaixo grava o núcleo do sistema como o módulo `rag_core.py`; as células seguintes o importam.
Assim o código-fonte (comentado) fica visível e autocontido no notebook, e o mesmo módulo é reusado pelos
scripts. *(A célula apenas escreve o arquivo, não treina nada.)*
"""))

cells.append(code("%%writefile rag_core.py" + chr(10) + RC_SRC))

cells.append(code("""
import rag_core as rc
print("Modelo de geração (local):", rc.LLM_MODEL)
print("Funções: rc.answer_baseline(q) | rc.answer_rag(q) | rc.answer_proposed(q)")
"""))

cells.append(md("""
## 3. Demonstração ao vivo
As 3 configurações sobre a mesma pergunta, um caso de domínio regional (paracoccidioidomicose),
onde o conhecimento paramétrico do modelo falha e a recuperação é decisiva. (A 1ª geração baixa e
carrega o modelo, ~6 GB; as seguintes são rápidas.)
"""))

cells.append(code("""
def rodar_demo(pergunta):
    print("PERGUNTA:", pergunta)
    for nome, fn in [("1) BASELINE: sem recuperação", rc.answer_baseline),
                     ("2) RAG: recuperação densa", rc.answer_rag),
                     ("3) PROPOSTA: densa + reranking", rc.answer_proposed)]:
        r = fn(pergunta)
        print("=" * 90)
        print(nome)
        print(r["answer"])
        if r["sources"]:
            print("Fontes:")
            for n, s in enumerate(r["sources"], 1):
                print(f"   [Fonte {n}] {s['title']}  ·  {s['disease']}")

# Exemplo 1 — doença regional (PCM): o conhecimento paramétrico falha, a recuperação é decisiva.
rodar_demo("Qual o tratamento de primeira escolha das formas leves a moderadas de PCM?")
"""))

cells.append(md("""
### Segundo exemplo (fibrose pulmonar): um detalhe factual
Outra pergunta, agora de fibrose pulmonar idiopática, um detalhe específico (os dois antifibróticos),
em que o baseline costuma errar ou omitir um dos nomes e a recuperação traz os dois com a fonte.
"""))

cells.append(code("""
# Exemplo 2 — detalhe específico (fibrose): os dois antifibróticos aprovados.
rodar_demo("Quais são os dois medicamentos antifibróticos aprovados para a fibrose pulmonar idiopática?")
"""))

cells.append(md("""
### O que o modelo realmente recebe
Na configuração RAG, o modelo não vê só a pergunta: recebe o prompt de sistema (persona) + os
trechos recuperados + a pergunta. A célula abaixo imprime esse prompt montado para a pergunta da demo,
tornando o mecanismo do RAG explícito.
"""))

cells.append(code('''
# Mostra o prompt REAL da config RAG: sistema (SYS_RAG) + TRECHOS recuperados + pergunta.
q_demo = "Qual o tratamento de primeira escolha das formas leves a moderadas de PCM?"
idxs = [i for i, _ in rc.dense_search(q_demo, 5)]
print("=== PROMPT DE SISTEMA (SYS_RAG) ===")
print(rc.SYS_RAG)
print()
print("=== MENSAGEM DE USUÁRIO (trechos recuperados + pergunta) ===")
user = f"TRECHOS:\\n{rc._context(idxs)}\\n\\nPergunta: {q_demo}\\n\\nResposta:"
print(user[:1500] + ("\\n[...]" if len(user) > 1500 else ""))
'''))

cells.append(md("""
## 4. Avaliação comparativa (54 perguntas)
O benchmark são 54 perguntas PT-BR que eu montei, de fácil a difícil, cada uma com uma
resposta-gabarito e a fonte-ouro de onde ela vem. Resultados pré-computados sobre esse conjunto
(rodar ao vivo leva ~30 min na GPU; `scripts/avaliar.py` reproduz). Métricas: qualidade da resposta
(LLM-as-judge, nota 0–2 vs. gabarito), fidelidade (a resposta se ancora nos trechos? padrão RAGAS),
taxa de citação de fontes, recuperação (hit-rate e MRR contra a fonte-ouro) e custo (latência,
tokens). *Hit-rate* = a fonte-ouro apareceu entre os trechos recuperados; *MRR* = quão perto do topo ela veio.
"""))

cells.append(code('''
# As 54 perguntas do benchmark: cada uma com pergunta, gabarito e a fonte-ouro de onde vem.
import json
try:
    import pandas as pd
    from IPython.display import display
    _PD = True
except Exception:
    _PD = False
bench = json.load(open("benchmark/perguntas.json", encoding="utf-8"))["perguntas"]
print(f"Total: {len(bench)} perguntas | por doenca:",
      {d: sum(q["disease"] == d for q in bench) for d in ("dpoc", "fibrose", "pcm")},
      "| por dificuldade:",
      {x: sum(q["dificuldade"] == x for q in bench) for x in ("facil", "media", "dificil")})
def _corta(s, n=90):
    s = s or ""
    return (s[:n] + "...") if len(s) > n else s
linhas = [{"id": q["id"], "doenca": q["disease"], "dif.": q["dificuldade"],
           "pergunta": _corta(q["pergunta"]), "gabarito": _corta(q["resposta_referencia"]),
           "fonte-ouro": q["fonte_id"]} for q in bench]
if _PD:
    display(pd.DataFrame(linhas).set_index("id"))
else:
    for r in linhas:
        print(r["id"], "|", r["doenca"], "|", r["dif."], "|", r["pergunta"])
'''))

cells.append(md("""
### O harness de avaliação (`scripts/avaliar.py`)
O código abaixo roda as 3 configurações em cada pergunta e mede tudo, usando o mesmo modelo local como
juiz (LLM-as-judge). Já foi rodado, os resultados prontos vêm no repositório (rodar de novo leva
~30 min na GPU). A célula grava o arquivo para deixar a lógica visível, mas não o executa aqui.
"""))

cells.append(code("%%writefile scripts/avaliar.py" + chr(10) + AVAL_SRC))

cells.append(md("""
### O juiz (LLM-as-judge): como a qualidade é medida
A qualidade é dada pelo mesmo modelo atuando como juiz, de forma pontual e cega: cada resposta é
comparada ao gabarito sem saber qual configuração a gerou (evita viés de posição). O juiz recebe duas
mensagens (system + user):

**Sistema** (`SYS_JUDGE`, em `rag_core.py`):
> *"Você é um avaliador médico rigoroso e objetivo. Siga exatamente as instruções de avaliação fornecidas a seguir."*

**Usuário** (a régua propriamente dita, montada em `judge_one` no `avaliar.py`):
> *"Compare a RESPOSTA ao GABARITO e atribua UMA nota: **2** = correta e suficiente (contém o ponto principal
> do gabarito, mesmo com texto extra); **1** = parcialmente correta (cita o ponto certo mas incompleta, ou
> com erro secundário); **0** = incorreta, vazia ou recusa. Responda APENAS em JSON: {"nota": 0, 1 ou 2}"*

A nota reportada é a média, reescalada para 0–1. A fidelidade usa a mesma mecânica (o juiz checa
se a resposta se apoia nos trechos recuperados). Como o juiz é o mesmo modelo que gera, pode haver viés
de auto-preferência; por isso a escala é grosseira (0/1/2, robusta a ruído) e as conclusões recomendam
conferência humana de um subconjunto.
"""))

cells.append(code("%%writefile scripts/resumo_resultados.py" + chr(10) + RESUMO_SRC))

cells.append(code("""
# tabela-resumo (geral + por doença + por dificuldade) a partir dos resultados pré-computados
%run scripts/resumo_resultados.py
"""))

cells.append(code('''
# figura comparativa (qualidade, fidelidade, hit-rate e latência)
from IPython.display import Image
import os
_fig = "_resultados/comparacao_all.png" if os.path.exists("_resultados/comparacao_all.png") else "_resultados/comparacao_final.png"
Image(filename=_fig)
'''))

cells.append(md("""
## 5b. Ablação de tamanho de modelo (3B vs 7B)
O achado honesto (o ganho da recuperação não chega à geração) foi observado no 3B. Para testar se
ele se propaga com escala, rodei a mesma avaliação num modelo maior (Qwen2.5-7B, em 4-bit), com a
mesma recuperação, só troca o gerador. Abaixo, a tabela completa do 7B (as mesmas métricas da §4)
e um gráfico 3B×7B: se a *Proposta* passar a superar o *RAG* com o 7B, o ganho do reranking se propaga
com escala.
"""))

cells.append(code('''
# Tabela COMPLETA do 7B (o mesmo resumo da §4, so troca o arquivo de resultados).
import os
os.environ["AVAL_FILE"] = "aval_all_7b.json"
%run scripts/resumo_resultados.py
os.environ["AVAL_FILE"] = "aval_all.json"   # restaura o default (3B)
'''))

cells.append(code('''
# Grafico da ablacao (3B vs 7B) no mesmo estilo da figura das 3 configuracoes.
# Barras agrupadas por modelo; o proposta (verde) so passa o RAG (azul) no 7B.
%matplotlib inline
import json, numpy as np, matplotlib.pyplot as plt
a3 = json.load(open("_resultados/aval_all.json", encoding="utf-8"))["agg"]
a7 = json.load(open("_resultados/aval_all_7b.json", encoding="utf-8"))["agg"]
cfgs = ["baseline", "rag", "proposta"]; labels = ["Baseline", "RAG", "Proposta"]
cols = ["#9aa0a6", "#4285F4", "#34A853"]; models = ["3B", "7B"]; aggs = [a3, a7]
x = np.arange(len(models)); w = 0.2
fig, ax = plt.subplots(1, 4, figsize=(18, 4.5))
def painel(ai, key, titulo, fmt, ymax):
    allvals = []
    for j, cfg in enumerate(cfgs):
        vals = [ag[cfg][key] for ag in aggs]
        allvals += vals
        bars = ai.bar(x + (j - 1) * w, vals, w, label=labels[j], color=cols[j])
        for b, v in zip(bars, vals):
            ai.annotate(fmt(v), (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=9)
    ai.set_title(titulo); ai.set_xticks(x)
    ai.set_xticklabels([f"Qwen2.5-{m}" for m in models])
    ai.set_ylim(0, ymax if ymax else max(allvals) * 1.18)
painel(ax[0], "qualidade", "Qualidade da resposta (0-1)", lambda v: f"{v:.2f}", 1.0)
painel(ax[1], "pct_correto", "Respostas corretas (%)", lambda v: f"{v:.0%}", 0.8)
painel(ax[2], "fidelidade", "Fidelidade (0-1)", lambda v: f"{v:.2f}", 1.0)
painel(ax[3], "lat_med", "Latencia media (s)", lambda v: f"{v:.0f}s", None)
ax[0].legend(loc="upper left")
fig.suptitle("Ablacao: o ganho do reranking (Proposta vs RAG) aparece no modelo maior", fontweight="bold")
fig.tight_layout(); plt.show()
'''))

cells.append(md("""
### Síntese numérica dos resultados
Os números-chave, lidos dos agregados pré-computados; é o que a conclusão abaixo interpreta.
"""))

cells.append(code(r'''
# Sintese numerica (lida dos agregados) — os numeros que a conclusao interpreta.
import json, re
def _load(fn):
    d = json.load(open(f"_resultados/{fn}", encoding="utf-8"))
    return d["agg"], d["rows"]
ag3, rows3 = _load("aval_all.json")
ag7, _ = _load("aval_all_7b.json")
def _cita(rows, c):
    return sum(bool(re.search(r"\[Fonte", r[c]["answer"] or "")) for r in rows) / len(rows)
print("SINTESE (54 perguntas, modelo 3B):")
print(f"  Qualidade (0-1):   baseline {ag3['baseline']['qualidade']:.2f}  ->  RAG {ag3['rag']['qualidade']:.2f}  ->  proposta {ag3['proposta']['qualidade']:.2f}")
print(f"  Fidelidade (0-1):  baseline {ag3['baseline']['fidelidade']:.2f}  ->  RAG {ag3['rag']['fidelidade']:.2f}  ->  proposta {ag3['proposta']['fidelidade']:.2f}")
print(f"  Cita a fonte:      baseline {_cita(rows3,'baseline'):.0%}  ->  RAG {_cita(rows3,'rag'):.0%}  ->  proposta {_cita(rows3,'proposta'):.0%}")
print(f"  Recuperacao:       hit-rate RAG {ag3['rag']['hit_rate']:.0%} -> proposta {ag3['proposta']['hit_rate']:.0%}   (MRR {ag3['rag']['mrr']:.2f} -> {ag3['proposta']['mrr']:.2f})")
print()
print("Ablacao de escala (Proposta vs RAG, mesma recuperacao nos dois):")
for tag, a in [("3B", ag3), ("7B", ag7)]:
    d = a["proposta"]["qualidade"] - a["rag"]["qualidade"]
    venc = "empata" if abs(d) < 0.02 else ("Proposta vence" if d > 0 else "RAG a frente")
    print(f"  {tag}: qualidade Proposta {a['proposta']['qualidade']:.2f} vs RAG {a['rag']['qualidade']:.2f} ({venc})"
          f"  |  correto Proposta {a['proposta']['pct_correto']:.0%} vs RAG {a['rag']['pct_correto']:.0%}")
'''))

cells.append(md("""
## 5. Conclusões, limitações e trabalhos futuros
> Este foi um estudo experimental de recuperação + geração sobre modelos pré-treinados
> (*transfer learning*): a contribuição não é treinar um modelo novo, mas a comparação controlada das
> 3 configurações e a técnica avançada (reranking), avaliadas com baselines e métricas casadas ao
> problema (recuperação, qualidade da resposta, fidelidade, citação de fontes, custo).

*(Os números abaixo são lidos das tabelas/figuras acima; conferir após cada re-execução.)*

- **A recuperação supera o baseline** na qualidade, com ganho concentrado nas doenças de menor cobertura
  paramétrica (fibrose, PCM); além disso, só as configurações com RAG citam a fonte e têm fidelidade
  alta, enquanto o baseline responde "de cabeça" e chega a alucinar (confundiu a sigla PCM), com fidelidade baixa.
  Rastreabilidade e fidelidade são essenciais em medicina.
- A **proposta** (densa + reranking) vence na recuperação (hit-rate/MRR): o cross-encoder cumpre o
  que promete, recuperando com mais precisão.
- **Achado principal (ablação de escala):** no 3B, o ganho de recuperação da proposta não se propaga
  para a geração (proposta ≈ RAG na qualidade), um LLM pequeno dilui o contexto mais rico. No 7B, com a
  mesma recuperação, o ganho aparece: a proposta passa a superar o RAG em qualidade e em respostas
  corretas (ver §5b). O reranking se traduz em respostas melhores quando o modelo é grande o suficiente.
- **Limitações**: o LLM-as-judge é o mesmo modelo (possível viés de auto-preferência), com conferência
  humana de um subconjunto; o hit-rate compara com uma fonte-ouro por pergunta (a KB tem documentos
  sobrepostos, então pode subestimar); reprodutível na mesma GPU (greedy/determinístico; entre hardwares, fp16
  pode variar pouco).
- **Futuro**: recuperação agêntica (multi-passo), métricas de fidelidade mais ricas e validação clínica com
  especialistas.

**Fontes e licenças:** `FONTES.md`. **Código:** todo embutido acima (`rag_core`, `indexar`, `avaliar`, `resumo`).
"""))

cells.append(md("""
## Referências técnicas
- **RAG:** Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS 2020.
- **Embeddings (e5):** Wang et al., *Text Embeddings by Weakly-Supervised Contrastive Pre-training*, 2022.
- **Reranking (cross-encoder):** Nogueira & Cho, *Passage Re-ranking with BERT*, 2019, reranker `BAAI/bge-reranker-v2-m3`.
- **HyDE:** Gao et al., *Precise Zero-Shot Dense Retrieval without Relevance Labels*, 2022.
- **Fidelidade (groundedness):** Es et al., *RAGAS: Automated Evaluation of Retrieval Augmented Generation*, 2023.
- **Modelo de geração:** Qwen2.5 (Qwen Team, 2024). **Busca vetorial:** FAISS (Johnson et al., 2019).
"""))

cells.append(md("""
## 6. Experimento (extra): HyDE (consulta por documento hipotético)
*Exploração adicional, isolada das 3 configurações acima (não altera baseline/RAG/proposta).*

Perguntas clínicas curtas ("O que é PCM?") casam mal com a redação dos trechos. O HyDE
(*Hypothetical Document Embeddings*, Gao et al., 2022) contorna isso: o próprio LLM redige um
documento hipotético (uma resposta plausível, em estilo de diretriz) e é *esse* texto, mais
próximo da linguagem do corpus, que vira a consulta densa. Abaixo medimos só o acerto da busca
(a fonte-ouro entra no contexto?) nas 54 perguntas, comparando a recuperação densa simples com a via HyDE.
Isso explora justamente a direção de *"recuperação aprimorada"* apontada nas conclusões.

> **Resultado:** no benchmark completo (54 perguntas) o HyDE empata com a recuperação densa
> (mesmo hit-rate, 30%); ele recupera trechos diferentes, mas não melhora o acerto global. O documento
> hipotético de um modelo de 3B às vezes desvia do tema, o que limita o ganho nesta base. Fica como
> exploração, não como melhoria adotada.
"""))

cells.append(code('''
# Experimento opcional: HyDE vs. recuperacao densa simples no benchmark COMPLETO (acerto da busca).
# Isolado — usa rc.dense_search (config RAG) e rc.dense_search_hyde; nao toca as 3 configs.
# O hit-rate nas 54 e pre-computado (rodar o HyDE nas 54 sao 54 geracoes de doc hipotetico, ~10 min);
# aqui exibimos esse resultado e geramos UM exemplo ao vivo do "documento hipotetico".
import json
hb = json.load(open("_resultados/hyde_bench.json", encoding="utf-8"))
n = hb["n"]
print(f"hit-rate no benchmark completo (n={n}):   densa {hb['densa_hits'] / n:.0%}   |   HyDE {hb['hyde_hits'] / n:.0%}")
print(f"ambos acertam em {hb['both']} perguntas; so a densa em {hb['densa_only']}; so o HyDE em {hb['hyde_only']}")
print("-> recuperam trechos diferentes, mas empatam no acerto global (sem ganho liquido).")

# exemplo ao vivo do 'documento hipotetico' que o HyDE escreve (e que vira a consulta densa):
bench = json.load(open("benchmark/perguntas.json", encoding="utf-8"))["perguntas"]
ex = bench[0]
print("\\nPergunta:", ex["pergunta"])
print("Documento hipotetico gerado:", rc.hyde_doc(ex["pergunta"])[:300])
'''))

nb = {"cells": cells, "metadata": {"language_info": {"name": "python"},
       "colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "nbformat": 4, "nbformat_minor": 5}

path = os.path.join(OUT, "Assistente_Pneumologia_RAG.ipynb")
json.dump(nb, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Notebook gerado: {path}  ({len(cells)} células)")

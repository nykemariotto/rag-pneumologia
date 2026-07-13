"""
rag_core.py - Nucleo do assistente de pneumologia (RAG).
Implementacao individual (Trabalho Final AM1PIDPG-93, UNESP) - autor: Nycolas Mariotto.

Reune a logica das 3 configuracoes comparadas no trabalho:
  - answer_baseline(q)  : LLM SEM consulta (so de cabeca)
  - answer_rag(q)       : busca densa (FAISS) top-k -> LLM                  [RAG convencional]
  - answer_proposed(q)  : busca densa amplia o pool -> reranker reordena -> LLM [proposta]

A GERACAO usa um modelo ABERTO rodando LOCALMENTE (HuggingFace transformers):
Qwen2.5-3B-Instruct (Apache-2.0, forte em portugues, cabe em ~6 GB fp16 e na GPU T4 do Colab).
Assim o notebook roda 100% no Colab com "Run all" e SEM chave de API. O modelo e configuravel
por ambiente (LLM_MODEL) e, em GPU pequena, LLM_4BIT=1 carrega um modelo MAIOR em 4-bit (usado
na ablacao de tamanho de modelo: 3B vs 7B).

Modelos carregados de forma "preguicosa" (lazy): so baixam/inicializam no 1o uso.
O notebook do Colab importa este mesmo modulo, garantindo que codigo testado == codigo entregue.
Geracao DETERMINISTICA (greedy, do_sample=False) -> resultados reprodutiveis.
"""
import os, re, json, pickle, unicodedata
import numpy as np

# Saida limpa no notebook: silencia avisos/barras de download do HuggingFace (HF_TOKEN opcional,
# barras de progresso, advisories) para nao poluir as respostas da demonstracao. (Definir ANTES
# de importar transformers/huggingface_hub, que sao carregados de forma preguicosa abaixo.)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import warnings, logging
warnings.filterwarnings("ignore")
for _n in ("huggingface_hub", "huggingface_hub.utils._http", "transformers"):
    logging.getLogger(_n).setLevel(logging.ERROR)

ROOT = os.path.dirname(os.path.abspath(__file__))
IDX = os.path.join(ROOT, "indices")
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
# Embedder/reranker na CPU por padrao: liberam VRAM para o prefill do LLM. Em placa de 8 GB, reranker
# na GPU rouba esse headroom -> geracao de contexto longo ~3x mais lenta (fp32 na CPU tambem da scores
# deterministicos). RERANK_DEVICE=cuda em GPU folgada.
EMB_DEVICE = os.environ.get("EMB_DEVICE", "cpu")
RERANK_DEVICE = os.environ.get("RERANK_DEVICE", "cpu")
# Modelo maior numa GPU pequena: LLM_4BIT=1 carrega em 4-bit (NF4) via bitsandbytes. O padrao (3B)
# usa fp16 e NAO depende de bitsandbytes. Ex.: LLM_MODEL=Qwen/Qwen2.5-7B-Instruct LLM_4BIT=1 (ablacao).
LOAD_4BIT = os.environ.get("LLM_4BIT", "").lower() in ("1", "true", "yes")

# --------------------------------------------------------------------------- #
# Carregamento preguicoso (lazy) de indices e modelos
# --------------------------------------------------------------------------- #
_state = {"emb": None, "rer": None, "faiss": None, "bm25": None, "chunks": None,
          "llm": None, "meta": None, "tok": 0}

def _meta():
    if _state["meta"] is None:
        _state["meta"] = json.load(open(os.path.join(IDX, "meta.json"), encoding="utf-8"))
    return _state["meta"]

def chunks():
    if _state["chunks"] is None:
        _state["chunks"] = [json.loads(l) for l in open(os.path.join(IDX, "chunks.jsonl"), encoding="utf-8")]
    return _state["chunks"]

def _faiss():
    if _state["faiss"] is None:
        import faiss
        _state["faiss"] = faiss.read_index(os.path.join(IDX, "faiss.index"))
    return _state["faiss"]

def _bm25():
    if _state["bm25"] is None:
        _state["bm25"] = pickle.load(open(os.path.join(IDX, "bm25.pkl"), "rb"))["bm25"]
    return _state["bm25"]

def _emb():
    if _state["emb"] is None:
        from sentence_transformers import SentenceTransformer
        _state["emb"] = SentenceTransformer(_meta()["emb_model"], device=EMB_DEVICE)
    return _state["emb"]

def _reranker():
    if _state["rer"] is None:
        import torch
        from sentence_transformers import CrossEncoder
        dev = RERANK_DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
        if dev == "cuda":
            _llm()  # carrega o LLM PRIMEIRO: em placa de 8 GB ele precisa ocupar a GPU inteira antes
                    # do reranker, senao 'device_map=auto' descarrega camadas p/ a CPU (geracao ~3x lenta)
        mk = {"torch_dtype": torch.float16} if dev == "cuda" else {}
        _state["rer"] = CrossEncoder(RERANKER_MODEL, max_length=512, device=dev, model_kwargs=mk)
    return _state["rer"]

def _llm():
    """Carrega o modelo aberto 1x (lazy). Padrao: GPU em fp16 (device_map='auto') ou CPU em fp32
    sem CUDA. Com LLM_4BIT=1, carrega em 4-bit (NF4) - permite rodar um modelo MAIOR (ex.: 7B) numa
    GPU pequena, ao custo de exigir 'bitsandbytes'. Precisa de 'accelerate' instalado."""
    if _state["llm"] is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        use_cuda = torch.cuda.is_available()
        tok = AutoTokenizer.from_pretrained(LLM_MODEL)
        kw = {"device_map": "auto"}
        if use_cuda and LOAD_4BIT:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        else:
            kw["dtype"] = torch.float16 if use_cuda else torch.float32  # 'dtype' (torch_dtype foi deprecado)
        model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, **kw)
        model.eval()
        _state["llm"] = (tok, model)
    return _state["llm"]

# --------------------------------------------------------------------------- #
# Busca (retrieval) - tudo local e deterministico, sem chave
# --------------------------------------------------------------------------- #
def _strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def _tokenize(text):
    return re.findall(r"[a-z0-9]+", _strip_accents(text.lower()))

def dense_search(query, k=10):
    """Busca por significado: embedding da pergunta (prefixo 'query:' do e5) vs FAISS."""
    q = _emb().encode([f"query: {query}"], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    D, I = _faiss().search(q, k)
    return [(int(i), float(d)) for d, i in zip(D[0], I[0]) if i >= 0]

def bm25_search(query, k=10):
    """Busca por palavra-chave (BM25)."""
    scores = _bm25().get_scores(_tokenize(query))
    order = np.argsort(-scores, kind="stable")[:k]  # estavel: empates por indice (reproduzivel)
    return [(int(i), float(scores[i])) for i in order]

def _rrf(rankings, k=60):
    """Reciprocal Rank Fusion: combina varias listas ordenadas em um ranking unico."""
    acc = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            acc[idx] = acc.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(acc, key=acc.get, reverse=True)

def hybrid_search(query, k=10, pool=20):
    """Combina busca densa + BM25 via RRF."""
    d = [i for i, _ in dense_search(query, pool)]
    b = [i for i, _ in bm25_search(query, pool)]
    return _rrf([d, b])[:k]

def rerank(query, cand_idx, top_n=5):
    """Reordena candidatos com cross-encoder (leitor criterioso)."""
    if not cand_idx:
        return []
    ch = chunks()
    pairs = [[query, ch[i]["text"]] for i in cand_idx]
    scores = np.asarray(_reranker().predict(pairs, batch_size=16))  # batch limitado: pico de VRAM seguro
    order = np.argsort(-scores, kind="stable")[:top_n]  # estavel: empates por indice
    return [(cand_idx[o], float(scores[o])) for o in order]

# --------------------------------------------------------------------------- #
# PROMPTS DE SISTEMA (a persona/instrucoes que cada configuracao passa ao modelo)
# --------------------------------------------------------------------------- #
# Persona forte e IGUAL no baseline e no RAG (justo: entre eles so a RECUPERACAO muda).
SYS_BASE = ("Você é um médico pneumologista experiente, com sólido domínio de medicina baseada "
            "em evidências. Responda à pergunta clínica de forma objetiva, tecnicamente precisa e "
            "concisa, em português do Brasil, priorizando o que é consenso na literatura. Se não "
            "tiver certeza, diga que não sabe em vez de inventar.")
SYS_RAG = ("Você é um médico pneumologista experiente, com sólido domínio de medicina baseada em "
           "evidências. Use PRIORITARIAMENTE os TRECHOS fornecidos e cite a fonte como [Fonte N] "
           "sempre que a informação vier deles. Se algum detalhe não estiver nos trechos, complemente "
           "com conhecimento clínico consolidado, deixando claro o que veio das fontes. Não invente "
           "fontes nem dados. Responda de forma objetiva, tecnicamente precisa e concisa, em "
           "português do Brasil.")
# Sistema do juiz = papel + rigor. A REGUA (notas 0/1/2, JSON) vai na mensagem de USUARIO, montada
# em avaliar.py (judge_one) -> e a ela que "as instrucoes fornecidas a seguir" se referem.
SYS_JUDGE = ("Você é um avaliador médico rigoroso e objetivo. Siga exatamente as instruções de "
             "avaliação fornecidas a seguir.")

def reset_tokens():
    _state["tok"] = 0

def get_tokens():
    return _state["tok"]

def _complete(system, user, max_new_tokens=None):
    """Geracao local DETERMINISTICA (greedy). Aplica o chat template (system+user) do modelo.
    Retorna (texto, n_tokens) onde n_tokens = prompt + resposta (proxy de custo).
    `max_new_tokens` permite respostas curtas (ex.: juiz que so devolve um numero)."""
    import torch
    tok, model = _llm()
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(next(model.parameters()).device)
    n_in = inputs["input_ids"].shape[1]
    mnt = MAX_NEW_TOKENS if max_new_tokens is None else max_new_tokens
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=mnt, do_sample=False,
                             pad_token_id=(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id))
    new = out[0][n_in:]
    text = tok.decode(new, skip_special_tokens=True).strip()
    return text, int(n_in + new.shape[0])

def _gen(system, user):
    text, n = _complete(system, user)
    _state["tok"] += n
    return text

def raw_generate(prompt, max_new_tokens=None):
    """Geracao sem contabilizar tokens (usada pela IA-juiza na avaliacao)."""
    return _complete(SYS_JUDGE, prompt, max_new_tokens=max_new_tokens)[0]

def _context(idxs):
    # [Fonte N] numerado por DOCUMENTO (nao por chunk): chunks do mesmo doc compartilham o numero,
    # entao a citacao [Fonte N] no texto bate 1:1 com a lista de fontes (deduplicada por doc) exibida.
    ch = chunks()
    num, blocks = {}, []
    for i in idxs:
        c = ch[i]
        d = c["doc_id"]
        if d not in num:
            num[d] = len(num) + 1
        blocks.append(f"[Fonte {num[d]}: {c['title']} — {c['disease']}]\n{c['text']}")
    return "\n\n".join(blocks)

def _sources(idxs):
    ch = chunks()
    seen, out = set(), []
    for i in idxs:
        key = ch[i]["doc_id"]
        if key not in seen:
            seen.add(key)
            out.append({"title": ch[i]["title"], "disease": ch[i]["disease"], "url": ch[i]["url"]})
    return out

# =========================================================================== #
# AS 3 CONFIGURACOES COMPARADAS NO TRABALHO
# O que muda entre elas e SO a selecao dos trechos que chegam ao MESMO LLM:
#   CONFIG 1 (answer_baseline) : LLM sozinho, SEM recuperacao (so conhecimento parametrico)
#   CONFIG 2 (answer_rag)      : busca densa (FAISS) top-k -> LLM                [RAG convencional]
#   CONFIG 3 (answer_proposed) : busca densa (pool) -> RERANKER reordena -> LLM  [proposta]
# =========================================================================== #

# ---- CONFIG 1: BASELINE (sem recuperacao) ----
def answer_baseline(question):
    """Config 1: sem consulta - o LLM responde so de cabeca (conhecimento parametrico)."""
    out = _gen(SYS_BASE, f"Pergunta: {question}\n\nResposta:")
    return {"answer": out, "sources": [], "context_idx": []}

# ---- CONFIG 2: RAG CONVENCIONAL (busca densa -> LLM) ----
def answer_rag(question, k=5, system=SYS_RAG):
    """Config 2 (RAG convencional): busca densa top-k -> LLM. Sem reranker.
    `system` default = SYS_RAG (config oficial); aceita um prompt alternativo (usado em experimentos)."""
    idxs = [i for i, _ in dense_search(question, k)]
    user = f"TRECHOS:\n{_context(idxs)}\n\nPergunta: {question}\n\nResposta:"
    return {"answer": _gen(system, user), "sources": _sources(idxs), "context_idx": idxs}

# ---- CONFIG 3: PROPOSTA (busca densa -> reranker -> LLM) ----
def answer_proposed(question, k=5, pool=40, system=SYS_RAG):
    """Config 3 (proposta): a busca densa amplia o pool de candidatos (top-`pool`) e um RERANKER
    cross-encoder (leitor criterioso) os reordena por RELEVANCIA real, ficando com os `k` melhores
    -> LLM. O reranker neural e a tecnica avancada alem do RAG simples: recuperacao mais PRECISA,
    com o trecho-chave no topo. (BM25/hibrido tambem foram implementados - ver hybrid_search -, mas
    injetavam trechos fora do tema que confundiam o modelo pequeno; a densa+reranker ficou mais limpa.)"""
    cand = [i for i, _ in dense_search(question, pool)]
    idxs = [i for i, _ in rerank(question, cand, top_n=k)]
    user = f"TRECHOS:\n{_context(idxs)}\n\nPergunta: {question}\n\nResposta:"
    return {"answer": _gen(system, user), "sources": _sources(idxs), "context_idx": idxs}

# --------------------------------------------------------------------------- #
# Experimento H1 (opcional) - HyDE: Hypothetical Document Embeddings (Gao et al., 2022)
# Isolado das 3 configuracoes oficiais; nao altera baseline/rag/proposta.
# --------------------------------------------------------------------------- #
SYS_HYDE = ("Voce e um pneumologista. Escreva UM paragrafo curto (2 a 4 frases), tecnico e direto, "
            "como se fosse um trecho de diretriz que responde a pergunta. Nao cite fontes nem use rodeios.")

def hyde_doc(question):
    """Gera o 'documento hipotetico': uma resposta plausivel em linguagem de diretriz. Esse texto fica
    mais proximo dos TRECHOS reais do que a pergunta curta, melhorando o casamento na busca densa."""
    return _gen(SYS_HYDE, f"Pergunta: {question}\n\nTrecho:")

def dense_search_hyde(query, k=10):
    """Busca densa usando o documento hipotetico como consulta. Ele e embedado no MESMO espaco dos
    documentos (prefixo 'passage:' do e5), pois faz papel de documento, nao de pergunta."""
    hd = hyde_doc(query)
    v = _emb().encode([f"passage: {hd}"], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    D, I = _faiss().search(v, k)
    return [(int(i), float(d)) for d, i in zip(D[0], I[0]) if i >= 0]

def answer_hyde(question, k=5, system=SYS_RAG):
    """Experimento H1 (HyDE): o modelo redige um documento hipotetico, ele vira a consulta densa, e os
    trechos recuperados vao ao LLM. Mesmo gerador/leitor das outras configuracoes, so muda a consulta."""
    idxs = [i for i, _ in dense_search_hyde(question, k)]
    user = f"TRECHOS:\n{_context(idxs)}\n\nPergunta: {question}\n\nResposta:"
    return {"answer": _gen(system, user), "sources": _sources(idxs), "context_idx": idxs}

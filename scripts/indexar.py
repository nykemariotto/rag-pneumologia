"""
Indexacao da knowledge base (Trabalho Final RAG - Pneumologia).

Le os .txt da knowledge_base/, corta em pedacos ("chunks"), gera os embeddings
(modelo multilingue) e monta os dois indices de busca:
  - FAISS  (busca por significado / densa)  -> indices/faiss.index
  - BM25   (busca por palavra-chave)        -> indices/bm25.pkl
Salva tambem indices/chunks.jsonl (texto + metadados por pedaco) e indices/meta.json.

Roda no .venv do projeto (tem sentence-transformers, faiss, rank-bm25).
O notebook do Colab reusa exatamente esta logica.
"""
import os, re, json, glob, pickle, unicodedata
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB = os.path.join(ROOT, "knowledge_base")
IDX = os.path.join(ROOT, "indices")
EMB_MODEL = os.environ.get("EMB_MODEL", "intfloat/multilingual-e5-base")
CHUNK_SIZE = 1100      # ~caracteres por pedaco
CHUNK_OVERLAP = 150

# ----------------------------- leitura + chunking -----------------------------
def read_doc(path):
    """Le um .txt da KB, separa o cabecalho de atribuicao (metadados) do corpo."""
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    meta = {"title": "", "publisher": "", "url": "", "license": "", "lang": "", "disease": ""}
    body = raw
    if raw.startswith("==="):
        head, _, body = raw.partition("-" * 70)
        for line in head.splitlines():
            if line.startswith("==="):
                meta["title"] = line.strip("= ").strip()
            for key, pat in (("publisher", "Fonte:"), ("url", "URL:"), ("license", "Licenca:")):
                if line.startswith(pat):
                    meta[key] = line.split(":", 1)[1].strip()
            if line.startswith("Doenca:"):
                seg = line.split(":", 1)[1]
                meta["disease"] = seg.split("|")[0].strip()
                if "Idioma:" in line:
                    meta["lang"] = line.split("Idioma:")[1].strip()
    meta["doc_id"] = os.path.splitext(os.path.basename(path))[0]
    if not meta["disease"]:
        meta["disease"] = os.path.basename(os.path.dirname(path))
    return meta, body.strip()

def split_paragraphs(text):
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Corta em pedacos respeitando paragrafos; janela com sobreposicao p/ paragrafos longos."""
    chunks, cur = [], ""
    for para in split_paragraphs(text):
        if len(para) > size:
            if cur:
                chunks.append(cur); cur = ""
            step = max(1, size - overlap)   # evita laco infinito se overlap >= size
            i = 0
            while i < len(para):
                chunks.append(para[i:i + size])
                i += step
        elif len(cur) + len(para) + 2 <= size:
            cur = (cur + "\n\n" + para) if cur else para
        else:
            if cur:
                chunks.append(cur)
            cur = para
    if cur:
        chunks.append(cur)
    return [c.strip() for c in chunks if len(c.strip()) > 60]

# ----------------------------- tokenizacao BM25 -------------------------------
# Reutiliza o MESMO tokenizador do rag_core para o indice e a consulta nunca divergirem.
import sys
sys.path.insert(0, ROOT)
from rag_core import _tokenize as tokenize, _strip_accents as strip_accents  # noqa: E402,F401

# ----------------------------- pipeline ---------------------------------------
def build():
    # determinismo do build (o indice gerado e enviado junto = versao canonica)
    import random
    random.seed(0); np.random.seed(0)
    try:
        import torch; torch.manual_seed(0)
    except Exception:
        pass
    os.makedirs(IDX, exist_ok=True)
    paths = [p for p in glob.glob(os.path.join(KB, "*", "*.txt"))]
    paths.sort()
    records = []
    for path in paths:
        meta, body = read_doc(path)
        for j, ch in enumerate(chunk_text(body)):
            records.append({
                "id": f"{meta['doc_id']}__{j}",
                "doc_id": meta["doc_id"], "disease": meta["disease"],
                "title": meta["title"], "url": meta["url"],
                "license": meta["license"], "lang": meta["lang"],
                "chunk_idx": j, "text": ch,
            })
    print(f"{len(paths)} documentos -> {len(records)} pedacos (chunks)")

    # embeddings (e5 exige prefixo 'passage:' nos documentos)
    from sentence_transformers import SentenceTransformer
    print(f"Carregando modelo de embeddings: {EMB_MODEL} ...")
    model = SentenceTransformer(EMB_MODEL)
    passages = [f"passage: {r['text']}" for r in records]
    emb = model.encode(passages, batch_size=32, show_progress_bar=True,
                       normalize_embeddings=True, convert_to_numpy=True).astype("float32")

    # FAISS (produto interno em vetores normalizados = cosseno)
    import faiss
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, os.path.join(IDX, "faiss.index"))

    # BM25
    from rank_bm25 import BM25Okapi
    tokenized = [tokenize(r["text"]) for r in records]
    bm25 = BM25Okapi(tokenized)
    with open(os.path.join(IDX, "bm25.pkl"), "wb") as f:
        pickle.dump({"bm25": bm25, "tokenized_len": len(tokenized)}, f)

    # chunks + metadados
    with open(os.path.join(IDX, "chunks.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    np.save(os.path.join(IDX, "embeddings.npy"), emb)
    meta = {"emb_model": EMB_MODEL, "n_docs": len(paths), "n_chunks": len(records),
            "dim": int(emb.shape[1]), "chunk_size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP}
    json.dump(meta, open(os.path.join(IDX, "meta.json"), "w"), indent=2)

    by_disease = {}
    for r in records:
        by_disease[r["disease"]] = by_disease.get(r["disease"], 0) + 1
    print("Chunks por doenca:", by_disease)
    print(f"Indices salvos em {IDX}")

if __name__ == "__main__":
    build()

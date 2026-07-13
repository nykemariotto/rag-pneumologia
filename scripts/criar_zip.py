"""
Cria Trabalho-Final-RAG.zip compativel com Linux/Colab (barras normais '/'),
excluindo o ambiente pesado (.venv), a chave e os PDFs originais.
(O Compress-Archive do PowerShell usa '\\' e quebra o unzip do Colab.)
"""
import os, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "Trabalho-Final-RAG.zip")
BASE = os.path.basename(ROOT)
SKIP_DIRS = {".venv", "__pycache__", "_originais", ".git", ".ipynb_checkpoints"}
SKIP_FILES = {
    ".groq_key", ".gemini_key", ".gemini_key.txt",
    # notas internas de preparo (nao sao entregaveis do projeto)
    "PLANO.md", "SUBMETER.md", "ROTEIRO_VIDEO.md", "NOTAS_DA_SESSAO.md", "ROTEIRO.docx", "DOCUMENTACAO.md",
    # scripts de rascunho / diagnostico (nao fazem parte do pacote entregue)
    "testar_busca.py", "testar_fix.py", "testar_gemini.py", "testar_groq.py",
    "testar_pipeline.py", "testar_retrieval_fix.py", "validar_fix.py",
    "diag.py", "ver_resultado.py",
    # scripts de empacotamento (nao precisam ir dentro do proprio pacote)
    "gerar_notebook.py", "criar_zip.py",
    # embeddings crus: usados so para CONSTRUIR o indice FAISS; o notebook le faiss.index (nao isto)
    "embeddings.npy",
    # variantes/backups de resultado (so o aval_all.json final + comparacao_final.png ficam)
    "aval_all_groq_ref.json", "aval_all_v1_diluido.json", "aval_all_v2_estrito.json",
    "comparacao_all.png",
    # experimentos de analise extra (H3 prompt ancorado / H4 fidelidade) — nao sao parte do pacote entregue
    "experimentos_extra.py", "experimentos_extra.json", "rejulgar_fidelidade.py",
}
SKIP_EXT = {".tmp", ".zip", ".log"}

if os.path.exists(OUT):
    os.remove(OUT)
n = 0
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for dp, dns, fns in os.walk(ROOT):
        dns[:] = [d for d in dns if d not in SKIP_DIRS]
        for fn in fns:
            if (fn in SKIP_FILES or os.path.splitext(fn)[1].lower() in SKIP_EXT
                    or ".bak" in fn or fn.startswith("~$")):   # ~$ = arquivo temporario/lock do Office
                continue
            full = os.path.join(dp, fn)
            rel = os.path.relpath(full, ROOT).replace(os.sep, "/")  # barras normais p/ Linux
            z.write(full, BASE + "/" + rel)
            n += 1
print(f"ZIP criado: {OUT}  ({os.path.getsize(OUT) // 1024 // 1024} MB, {n} arquivos)")

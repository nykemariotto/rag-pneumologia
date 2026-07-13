"""Valida a correcao da proposta em 5 perguntas (compara consulta simples x melhorada)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rag_core as rc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
bm = {q["id"]: q for q in json.load(open(os.path.join(ROOT, "benchmark", "perguntas.json"), encoding="utf-8"))["perguntas"]}
IDS = ["dpoc_02", "dpoc_15", "fib_03", "pcm_05", "pcm_15"]
ch = rc.chunks()
for i in IDS:
    q = bm[i]
    print("=" * 95 + f"\n{i} | {q['pergunta']}", flush=True)
    print("GABARITO:", q["resposta_referencia"], flush=True)
    for name, fn in [("simples ", rc.answer_rag), ("melhorada", rc.answer_proposed)]:
        r = fn(q["pergunta"])
        docs = list(dict.fromkeys(ch[j]["doc_id"] for j in r["context_idx"]))
        print(f"-- {name}  docs={docs}", flush=True)
        print("   " + " ".join(r["answer"].split())[:330], flush=True)

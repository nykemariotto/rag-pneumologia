"""Valida a estrutura do benchmark: JSON ok, contagens por doenca/dificuldade,
e se cada fonte_id citada existe na knowledge base."""
import os, json, glob
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
bm = json.load(open(os.path.join(ROOT, "benchmark", "perguntas.json"), encoding="utf-8"))
qs = bm["perguntas"]
kb_ids = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(ROOT, "knowledge_base", "*", "*.txt"))}

print(f"Total de perguntas: {len(qs)}")
print("Por doenca:    ", dict(Counter(q["disease"] for q in qs)))
print("Por dificuldade:", dict(Counter(q["dificuldade"] for q in qs)))

ids = [q["id"] for q in qs]
assert len(ids) == len(set(ids)), "IDs duplicados!"
faltando = sorted({q["fonte_id"] for q in qs} - kb_ids)
print("fonte_id sem texto na estante:", faltando if faltando else "nenhum (todas as fontes existem)")
sem_resp = [q["id"] for q in qs if not q.get("resposta_referencia")]
print("Perguntas sem gabarito:", sem_resp if sem_resp else "nenhuma")
print("Fontes distintas usadas:", len({q["fonte_id"] for q in qs}))

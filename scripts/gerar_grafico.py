"""
Gera a figura final dos resultados (linguagem simples, sem jargao) a partir de aval_all.json:
  (1) Qualidade geral   (2) Qualidade por doenca   (3) Cita a fonte?   (4) Acerto da busca
Salva em _resultados/comparacao_final.png.
"""
import os, json, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = json.load(open(os.path.join(ROOT, "_resultados", "aval_all.json"), encoding="utf-8"))["rows"]
CFG = ["baseline", "rag", "proposta"]
NOME = ["Sem\nconsulta", "Consulta\nsimples", "Consulta\nmelhorada"]
COR = ["#9aa0a6", "#4285F4", "#34A853"]

def qual(sub, c):
    return sum(r[c]["nota"] for r in sub) / (2 * len(sub)) if sub else 0
def cita(c):
    return sum(bool(re.search(r"\[Fonte", r[c]["answer"] or "")) for r in rows) / len(rows)
def hit(c):
    return sum(r[c]["hit"] for r in rows) / len(rows)

fig, ax = plt.subplots(2, 2, figsize=(13, 9))

# (1) Qualidade geral
g = [qual(rows, c) for c in CFG]
b = ax[0, 0].bar(NOME, g, color=COR); ax[0, 0].set_ylim(0, 1)
ax[0, 0].set_title("Qualidade da resposta (0 a 1)\nmédia nas 54 perguntas", fontweight="bold")
for p in b: ax[0, 0].annotate(f"{p.get_height():.2f}", (p.get_x()+p.get_width()/2, p.get_height()), ha="center", va="bottom")

# (2) Qualidade por doenca
DIS = [("dpoc", "DPOC"), ("fibrose", "Fibrose"), ("pcm", "Paracoccidioido-\nmicose")]
x = np.arange(len(DIS)); w = 0.26
for j, c in enumerate(CFG):
    vals = [qual([r for r in rows if r["disease"] == d], c) for d, _ in DIS]
    ax[0, 1].bar(x + (j-1)*w, vals, w, label=NOME[j].replace("\n", " "), color=COR[j])
ax[0, 1].set_xticks(x); ax[0, 1].set_xticklabels([n for _, n in DIS]); ax[0, 1].set_ylim(0, 1)
ax[0, 1].set_title("Qualidade por doença\n(o ganho do RAG é maior na fibrose)", fontweight="bold")
ax[0, 1].legend(fontsize=8)

# (3) Cita a fonte?
cc = [cita(c) for c in CFG]
b = ax[1, 0].bar(NOME, cc, color=COR); ax[1, 0].set_ylim(0, 1)
ax[1, 0].set_title("Mostra de qual fonte tirou a resposta\n(essencial em medicina)", fontweight="bold")
for p in b: ax[1, 0].annotate(f"{p.get_height():.0%}", (p.get_x()+p.get_width()/2, p.get_height()), ha="center", va="bottom")

# (4) Acerto da busca (so as versoes com consulta)
hh = [hit("rag"), hit("proposta")]
b = ax[1, 1].bar(["Consulta\nsimples", "Consulta\nmelhorada"], hh, color=COR[1:]); ax[1, 1].set_ylim(0, 1)
ax[1, 1].set_title("Achou a fonte certa\n(entre os trechos consultados)", fontweight="bold")
for p in b: ax[1, 1].annotate(f"{p.get_height():.0%}", (p.get_x()+p.get_width()/2, p.get_height()), ha="center", va="bottom")

fig.suptitle("Assistente de Pneumologia: comparação das 3 versões (54 perguntas)", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(ROOT, "_resultados", "comparacao_final.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print("Figura salva:", out)

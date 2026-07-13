"""
Resumo dos resultados da avaliacao: legenda + geral + por doenca + por dificuldade.

COMO A NOTA E CALCULADA:
O LLM-as-judge compara CADA resposta com o gabarito e da uma nota bruta:
  0 = incorreta, vazia ou recusa  |  1 = parcialmente correta  |  2 = correta e suficiente.
A coluna 'qualidade (0-1)' e a MEDIA dessas notas, normalizada (dividida por 2) para a escala 0 a 1.
  'correto (%)'    = fracao de respostas que receberam a nota maxima (2).
  'cita fonte (%)' = fracao de respostas que citam a fonte no formato [Fonte N] (so as configs com recuperacao).
  'hit-rate (%)' / 'MRR' = a fonte-ouro entrou nos trechos recuperados, e quao perto do topo.
  'tempo (s)' / 'tokens' = custo medio por pergunta.
"""
import os, json, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Arquivo de resultados (default = 3B); AVAL_FILE=aval_all_7b.json reusa esta mesma tabela na ablacao.
AVAL_FILE = os.environ.get("AVAL_FILE", "aval_all.json")
rows = json.load(open(os.path.join(ROOT, "_resultados", AVAL_FILE), encoding="utf-8"))["rows"]

CFGS = ["baseline", "rag", "proposta"]
NAMES = {"baseline": "Baseline", "rag": "RAG convencional", "proposta": "Proposta"}
SHORT = {"baseline": "Baseline", "rag": "RAG", "proposta": "Proposta"}
DIS = {"dpoc": "DPOC", "fibrose": "Fibrose", "pcm": "PCM"}
DIF = {"facil": "Facil", "media": "Media", "dificil": "Dificil"}

def cita(ans): return bool(re.search(r"\[Fonte", ans or ""))
def qual(sub, c): return sum(r[c]["nota"] for r in sub) / (2 * len(sub)) if sub else 0.0

# ---- monta as tabelas (lista de dicts) ----
geral = []
for c in CFGS:
    notas = [r[c]["nota"] for r in rows]
    d = {"Versao": NAMES[c],
         "qualidade (0-1)": round(sum(notas) / (2 * len(notas)), 2),
         "fidelidade (0-1)": round(sum(r[c].get("fidelidade", 0) for r in rows) / (2 * len(rows)), 2),
         "correto (%)": round(100 * sum(n == 2 for n in notas) / len(notas)),
         "cita fonte (%)": round(100 * sum(cita(r[c]["answer"]) for r in rows) / len(rows)),
         "hit-rate (%)": "-", "MRR": "-",
         "tempo (s)": round(sum(r[c]["lat"] for r in rows) / len(rows)),
         "tokens": round(sum(r[c]["tok"] for r in rows) / len(rows))}
    if c in ("rag", "proposta"):
        d["hit-rate (%)"] = round(100 * sum(r[c]["hit"] for r in rows) / len(rows))
        d["MRR"] = round(sum((1 / r[c]["rank"]) for r in rows if r[c]["rank"] > 0) / len(rows), 2)
    geral.append(d)

def tabela_por(campo, mapa, collabel):
    out = []
    for k, nome in mapa.items():
        sub = [r for r in rows if r[campo] == k]
        if not sub:
            continue
        row = {collabel: nome, "n": len(sub)}
        for c in CFGS:
            row[SHORT[c]] = round(qual(sub, c), 2)
        out.append(row)
    return out

por_doenca = tabela_por("disease", DIS, "Doenca")
por_dif = tabela_por("dificuldade", DIF, "Dificuldade")

# ---- exibe: pandas (tabela bonita no Colab) ou texto ----
try:
    import pandas as pd
    from IPython.display import display
    USE_PD = True
except Exception:
    USE_PD = False

print(f"Metricas (calculadas sobre as {len(rows)} perguntas):")
print("  qualidade (0-1)  : media das notas do LLM-as-judge, reescalada para 0-1 (notas brutas: 0 = incorreta, 1 = parcial, 2 = correta)")
print("  fidelidade (0-1) : a resposta se ancora nos trechos recuperados? (groundedness, padrao RAGAS; baseline sem trechos = 0)")
print("  correto (%)      : fracao de respostas com a nota maxima (2)")
print("  cita fonte (%)   : fracao de respostas que citam a fonte como [Fonte N]")
print("  hit-rate / MRR   : com que frequencia (e quao perto do topo) a fonte-ouro entra nos trechos recuperados")
print()

def show(titulo, tabela, indice):
    print(f"=== {titulo} ===")
    if USE_PD:
        display(pd.DataFrame(tabela).set_index(indice))
    else:
        cols = list(tabela[0].keys())
        w = {c: max(len(str(c)), max(len(str(r[c])) for r in tabela)) for c in cols}
        print("  " + "  ".join(str(c).ljust(w[c]) for c in cols))
        for r in tabela:
            print("  " + "  ".join(str(r[c]).ljust(w[c]) for c in cols))
    print()

show(f"GERAL ({len(rows)} perguntas)", geral, "Versao")
show("QUALIDADE (0-1) POR DOENCA", por_doenca, "Doenca")
show("QUALIDADE (0-1) POR DIFICULDADE", por_dif, "Dificuldade")

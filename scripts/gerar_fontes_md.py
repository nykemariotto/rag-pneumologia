"""Gera FONTES.md (referencias + licencas) a partir de kb_sources.json."""
import os, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = json.load(open(os.path.join(ROOT, "kb_sources.json"), encoding="utf-8"))
DNOME = {"dpoc": "DPOC", "fibrose": "Fibrose pulmonar", "pcm": "Paracoccidioidomicose"}

L = ["# Fontes da Knowledge Base: Referencias e Licencas\n",
     f"Dominio: {cfg['domain']}\n",
     "Todas as fontes abaixo sao de acesso aberto e redistribuiveis (dominio publico ou "
     "Creative Commons). Cada documento mantem a atribuicao da fonte no cabecalho do arquivo .txt.\n"]

infold = [s for s in cfg["in_folder_sources"] if s.get("in_folder")]
for dis in ("dpoc", "fibrose", "pcm"):
    itens = [s for s in infold if s["disease"] == dis]
    L.append(f"\n## {DNOME[dis]} ({len(itens)} fontes)\n")
    for s in itens:
        lang = "PT" if s.get("lang") == "pt" else "EN"
        L.append(f"- **{s['title']}** · {s.get('publisher','')}. "
                 f"_Licenca: {s['license']} · {lang}_. <{s['url']}>")

L.append("\n## Referencias citadas (nao redistribuidas na pasta, por direitos autorais)\n")
for s in cfg.get("cite_only_sources", []):
    L.append(f"- **{s['title']}** · _{s['license']}_. <{s['url']}>")

L.append(f"\n---\nTotal na estante: {len(infold)} documentos redistribuiveis.\n")
open(os.path.join(ROOT, "FONTES.md"), "w", encoding="utf-8").write("\n".join(L))
print(f"FONTES.md gerado: {len(infold)} fontes na estante + {len(cfg.get('cite_only_sources', []))} citadas.")

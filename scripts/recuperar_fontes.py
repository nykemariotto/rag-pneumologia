"""
Recupera as fontes que falharam no coletor (bloqueio 403 em OUP/MDPI),
tentando rotas alternativas e abertas:
  1) PMC via DOI (E-utilities) -> texto limpo via API BioC
  2) Unpaywall (best OA location) -> PDF -> PyMuPDF
Se nenhuma funcionar, a fonte fica como 'cite-only' (referenciada, nao no folder).
"""
import json, os, re, time
import requests, fitz
from urllib.parse import urlparse
from bs4 import BeautifulSoup

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB = os.path.join(ROOT, "knowledge_base")
EMAIL = "n.mariotto@unesp.br"
BROWSER = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,pt-BR;q=0.8",
}
# DOIs das fontes que falharam
DOI = {
    "ats_pharm_copd_2020": "10.1164/rccm.202003-0625ST",
    "ipf_guideline_2022":  "10.1164/rccm.202202-0399ST",
    "mdpi_jfungi_pulm":    "10.3390/jof9020218",
}

def clean(text):
    lines = [l.strip() for l in text.splitlines()]
    out, blank = [], False
    for l in lines:
        if l: out.append(l); blank = False
        elif not blank: out.append(""); blank = True
    return "\n".join(out).strip()

def pmc_from_doi(doi):
    r = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                     params={"db": "pmc", "term": f"{doi}[DOI]", "retmode": "json"},
                     headers=BROWSER, timeout=60)
    ids = r.json().get("esearchresult", {}).get("idlist", [])
    return "PMC" + ids[0] if ids else None

def bioc(pmcid):
    api = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode"
    data = requests.get(api, headers=BROWSER, timeout=90).json()
    cols = data if isinstance(data, list) else [data]
    parts = [ (p.get("text") or "").strip()
              for c in cols for d in c.get("documents", []) for p in d.get("passages", []) ]
    return clean("\n\n".join([p for p in parts if p]))

def unpaywall_pdf(doi):
    j = requests.get(f"https://api.unpaywall.org/v2/{doi}", params={"email": EMAIL}, timeout=60).json()
    loc = j.get("best_oa_location") or {}
    return loc.get("url_for_pdf") or loc.get("url")

def save(src, text):
    d = os.path.join(KB, src["disease"]); os.makedirs(d, exist_ok=True)
    head = (f"=== {src['title']} ===\nFonte: {src.get('publisher','')}\nURL: {src['url']}\n"
            f"Licenca: {src['license']}\nDoenca: {src['disease']} | Idioma: {src.get('lang','')}\n" + "-"*70 + "\n\n")
    with open(os.path.join(d, src["id"] + ".txt"), "w", encoding="utf-8") as f:
        f.write(head + text + "\n")

def main():
    cfg = json.load(open(os.path.join(ROOT, "kb_sources.json"), encoding="utf-8"))
    byid = {s["id"]: s for s in cfg["in_folder_sources"]}
    for sid, doi in DOI.items():
        src = byid[sid]; text = ""
        # rota 1: PMC via DOI -> BioC
        try:
            pmcid = pmc_from_doi(doi)
            if pmcid:
                text = bioc(pmcid)
                print(f"  {sid}: PMC {pmcid} -> {len(text)} ch")
        except Exception as e:
            print(f"  {sid}: PMC falhou ({type(e).__name__})")
        # rota 2: Unpaywall PDF
        if len(text) < 800:
            try:
                pdf_url = unpaywall_pdf(doi)
                if pdf_url:
                    raw = requests.get(pdf_url, headers=BROWSER, timeout=120).content
                    od = os.path.join(KB, src["disease"], "_originais"); os.makedirs(od, exist_ok=True)
                    p = os.path.join(od, sid + ".pdf")
                    if not raw.startswith(b"%PDF"):
                        raise ValueError("conteudo retornado nao e PDF")
                    with open(p, "wb") as f:
                        f.write(raw)
                    doc = fitz.open(p); text = clean("\n".join(pg.get_text() for pg in doc)); doc.close()
                    print(f"  {sid}: Unpaywall PDF -> {len(text)} ch")
                else:
                    print(f"  {sid}: Unpaywall sem OA location")
            except Exception as e:
                print(f"  {sid}: Unpaywall falhou ({type(e).__name__}: {str(e)[:60]})")
        if len(text) >= 800:
            save(src, text); print(f"  {sid}: SALVO ({len(text)} ch)")
        else:
            print(f"  {sid}: nao recuperado -> cite-only")
        time.sleep(0.8)

if __name__ == "__main__":
    main()

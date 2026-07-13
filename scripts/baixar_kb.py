"""
Coletor da knowledge base (Trabalho Final RAG - Pneumologia).

Le kb_sources.json e baixa cada fonte redistribuivel (in_folder=true),
extraindo TEXTO LIMPO para knowledge_base/<doenca>/<id>.txt com cabecalho
de atribuicao (titulo, fonte, URL, licenca) para rastreabilidade.

- PDF  -> baixa original + extrai texto com PyMuPDF (fitz)
- PMC  -> API BioC (texto limpo, sem HTML); fallback p/ HTML
- NCBI Bookshelf (StatPearls) e demais HTML -> extracao com BeautifulSoup

Roda com o env 'geral' (tem fitz, requests, bs4). Idempotente: pula o que ja existe.
"""
import json, os, re, time, sys
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import fitz  # PyMuPDF

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB = os.path.join(ROOT, "knowledge_base")
SOURCES = os.path.join(ROOT, "kb_sources.json")
HEADERS = {"User-Agent": "Mozilla/5.0 (academic RAG course project; UNESP; contact n.mariotto@unesp.br)"}

def get(url, timeout=60):
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r

def clean_lines(text):
    lines = [ln.strip() for ln in text.splitlines()]
    out, blank = [], False
    for ln in lines:
        if ln:
            out.append(ln); blank = False
        elif not blank:
            out.append(""); blank = True
    return "\n".join(out).strip()

def extract_html(html, url):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "button", "noscript"]):
        tag.decompose()
    host = urlparse(url).netloc
    main = None
    if "ncbi.nlm.nih.gov" in host:
        main = soup.find(id="maincontent") or soup.find("div", class_=re.compile("body-content")) or soup.find("article")
    elif "pmc.ncbi.nlm.nih.gov" in host:
        main = soup.find("section", class_="body") or soup.find("article") or soup.find("main")
    elif "who.int" in host:
        main = soup.find("article") or soup.find("main")
    elif "cdc.gov" in host:
        main = soup.find("main") or soup.find(id="content")
    elif "medlineplus.gov" in host:
        main = soup.find(id="mplus-content") or soup.find("article") or soup.find("main")
    elif "wikipedia.org" in host:
        main = soup.find(id="mw-content-text")
    elif "oup.com" in host:
        main = soup.find("div", class_=re.compile("article-body|widget-items|abstract")) or soup.find("article") or soup.find("main")
    if main is None:
        # fallback: maior bloco de texto
        cands = soup.find_all(["main", "article", "div", "section"])
        main = max(cands, key=lambda t: len(t.get_text()), default=soup.body or soup)
    return clean_lines(main.get_text(separator="\n"))

def extract_pdf(path):
    doc = fitz.open(path)
    txt = "\n".join(page.get_text() for page in doc)
    doc.close()
    return clean_lines(txt)

def fetch_pmc_bioc(pmcid):
    api = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode"
    data = get(api, timeout=90).json()
    cols = data if isinstance(data, list) else [data]
    parts = []
    for col in cols:
        for doc in col.get("documents", []):
            for pas in doc.get("passages", []):
                t = (pas.get("text") or "").strip()
                if t:
                    parts.append(t)
    return clean_lines("\n\n".join(parts))

def header(s):
    return (f"=== {s['title']} ===\n"
            f"Fonte: {s.get('publisher','')}\n"
            f"URL: {s['url']}\n"
            f"Licenca: {s['license']}\n"
            f"Doenca: {s['disease']} | Idioma: {s.get('lang','')}\n"
            + "-" * 70 + "\n\n")

def process(s):
    disease_dir = os.path.join(KB, s["disease"])
    orig_dir = os.path.join(disease_dir, "_originais")
    os.makedirs(orig_dir, exist_ok=True)
    out_txt = os.path.join(disease_dir, s["id"] + ".txt")
    if os.path.exists(out_txt) and os.path.getsize(out_txt) > 500:
        return ("skip", os.path.getsize(out_txt))

    text = ""
    if s["type"] == "pdf":
        raw = get(s["url"], timeout=120).content
        if not raw.startswith(b"%PDF"):           # servidor pode devolver HTML de erro com 200
            return ("FALHA(nao-pdf)", len(raw))
        pdf_path = os.path.join(orig_dir, s["id"] + ".pdf")
        tmp = pdf_path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(raw)
        os.replace(tmp, pdf_path)
        text = extract_pdf(pdf_path)
    else:
        host = urlparse(s["url"]).netloc
        if "pmc.ncbi.nlm.nih.gov" in host:
            m = re.search(r"PMC\d+", s["url"])
            try:
                text = fetch_pmc_bioc(m.group(0)) if m else ""
            except Exception as e:
                text = ""
            if len(text) < 800:  # fallback HTML
                try:
                    text = extract_html(get(s["url"]).text, s["url"])
                except Exception:
                    pass
        else:
            text = extract_html(get(s["url"]).text, s["url"])

    text = (text or "").strip()
    if len(text) < 400:
        return ("FALHA(curto)", len(text))
    tmp_txt = out_txt + ".tmp"
    with open(tmp_txt, "w", encoding="utf-8") as f:
        f.write(header(s) + text + "\n")
    os.replace(tmp_txt, out_txt)               # escrita atomica (resume nao pega arquivo truncado)
    return ("ok", len(text))

def main():
    with open(SOURCES, encoding="utf-8") as f:
        cfg = json.load(f)
    sources = [s for s in cfg["in_folder_sources"] if s.get("in_folder")]
    report, total_chars = {}, 0
    print(f"Coletando {len(sources)} fontes...\n")
    for s in sources:
        try:
            status, n = process(s)
        except Exception as e:
            status, n = f"ERRO: {type(e).__name__}: {str(e)[:80]}", 0
        report[s["id"]] = {"status": status, "chars": n, "disease": s["disease"], "title": s["title"]}
        if status in ("ok", "skip"):
            total_chars += n
        print(f"  [{status:>14}] {n:>7} ch  {s['disease']:<7} {s['id']}")
        time.sleep(0.6)
    with open(os.path.join(ROOT, "kb_build_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    pages = total_chars / 3000  # ~3000 ch/pagina
    print(f"\nTotal: {total_chars:,} caracteres  (~{pages:.0f} paginas estimadas)")
    fails = [k for k, v in report.items() if v["status"] not in ("ok", "skip")]
    if fails:
        print(f"Falhas ({len(fails)}): {', '.join(fails)}")

if __name__ == "__main__":
    main()

"""
Avaliacao comparativa das 3 configuracoes (baseline / RAG / proposta).
Para cada pergunta: roda as 3 versoes e mede
  - qualidade (IA-juiza compara cada resposta ao gabarito: nota 0/1/2)
  - fidelidade (groundedness, padrao RAGAS: a resposta se ancora nos trechos recuperados?)
  - acerto da busca (a fonte certa entrou no contexto? -> hit-rate e MRR) [rag e proposta]
  - custo (latencia REAL em s, tokens prompt+resposta do modelo local)
GERACAO E JUIZA usam o MESMO modelo aberto LOCAL (Qwen2.5 via transformers) - sem chave, sem cota,
deterministico (greedy). RESUMIVEL: salva o progresso a cada pergunta em _resultados/aval_<modo>.json;
um crash no meio continua de onde parou. Gera grafico no final. O modelo e configuravel por ambiente
(LLM_MODEL / LLM_4BIT); MODEL_TAG separa os arquivos de saida (ex.: 3b vs 7b na ablacao).

Uso:  python avaliar.py all   (54 perguntas)  |  python avaliar.py preview  (12)
"""
import os, sys, json, time, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rag_core as rc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "_resultados")
os.makedirs(OUT, exist_ok=True)

PREVIEW = ["dpoc_02", "dpoc_08", "dpoc_10", "dpoc_13", "fib_06", "fib_07",
           "fib_09", "fib_13", "pcm_04", "pcm_08", "pcm_11", "pcm_14"]
CFGS = [("baseline", rc.answer_baseline), ("rag", rc.answer_rag), ("proposta", rc.answer_proposed)]

def _save(path, obj):
    """Escrita atomica (.tmp -> replace): um crash no meio nao corrompe o checkpoint."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _parse_nota(raw):
    """Extrai a nota 0/1/2 da resposta da juiza. ANCORA em 'nota' (cobre {"nota": 2}, 'Nota 1',
    'nota=0'); so se nao houver, usa o ULTIMO 0/1/2 isolado (o veredito costuma vir por ultimo) -
    evita pegar um digito do raciocinio/conteudo (ex.: '2 criterios', '0,70')."""
    raw = raw or ""
    ms = re.findall(r'nota["\s:=]*([012])', raw, re.IGNORECASE)  # ULTIMO match: o veredito vem por
    if ms:                                                        # ultimo, apos qualquer eco do molde
        return int(ms[-1])
    achados = re.findall(r"\b([012])\b", raw)
    return int(achados[-1]) if achados else None

def judge_one(pergunta, gabarito, answer):
    """Avalia UMA resposta vs gabarito (pontual, CEGA a qual configuracao gerou -> sem vies de
    posicao). Cada resposta e julgada pelo seu proprio merito contra a referencia."""
    p = ("Você é um avaliador médico rigoroso. Compare a RESPOSTA ao GABARITO e atribua UMA nota: "
         "2 = correta e suficiente (contém o ponto principal do gabarito, mesmo que com texto extra); "
         "1 = parcialmente correta (cita o ponto certo mas incompleta, ou com erro secundário); "
         "0 = incorreta, vazia ou recusa a responder.\n\n"
         f"PERGUNTA: {pergunta}\nGABARITO: {gabarito}\n\nRESPOSTA:\n{answer}\n\n"
         'Responda APENAS em JSON: {"nota": 0, 1 ou 2}')
    nota = _parse_nota(rc.raw_generate(p))
    if nota is None:
        print("  [aviso] juiza nao retornou nota valida; assumindo 0", flush=True)
        return 0
    return nota

def judge_faithfulness(answer, context):
    """FIDELIDADE (groundedness, padrao RAGAS): a RESPOSTA usa a informacao dos TRECHOS recuperados?
    Formulacao simples e direta -- a versao rigida ('APENAS em JSON' + regua dura) colapsava o juiz
    de 3B para 0 ate em respostas ancoradas. Nota 0/1/2. Sem contexto (baseline) -> 0."""
    if not (context or "").strip():
        return 0
    p = ("Os TRECHOS abaixo foram fornecidos a um assistente médico. A RESPOSTA usa a informação "
         "desses trechos? 2 = a maior parte da resposta se apoia nos trechos; 1 = em parte; "
         "0 = a resposta ignora os trechos ou os contradiz.\n\n"
         f"TRECHOS:\n{context}\n\nRESPOSTA:\n{answer}\n\n"
         "Responda só com o número (0, 1 ou 2). Nota:")
    n = _parse_nota(rc.raw_generate(p))
    return 0 if n is None else n

def load_done(path):
    if os.path.exists(path):
        try:
            return {r["id"]: r for r in json.load(open(path, encoding="utf-8")).get("rows", [])}
        except Exception:
            return {}
    return {}

def run(qs, path):
    ch = rc.chunks()
    done = load_done(path)
    rows = [done[i] for i in done]
    if len(done) < len(qs):  # aquece modelos (LLM+embedder+reranker) p/ latencias limpas (sem o load 1x)
        print("  aquecendo modelos (1a carga)...", flush=True)
        rc.answer_proposed("aquecimento do sistema")
    for q in qs:
        if q["id"] in done:
            print(f"  (ja feito) {q['id']}")
            continue
        item = {"id": q["id"], "disease": q["disease"], "dificuldade": q["dificuldade"], "fonte_id": q["fonte_id"]}
        ans = {}
        for cfg, fn in CFGS:
            rc.reset_tokens()
            t0 = time.time()
            r = fn(q["pergunta"])
            lat = time.time() - t0
            tok = rc.get_tokens()
            docs = [ch[i]["doc_id"] for i in r["context_idx"]]
            uniq = list(dict.fromkeys(docs))            # docs unicos na ordem -> MRR por doc, nao por chunk
            hit = q["fonte_id"] in uniq
            rank = uniq.index(q["fonte_id"]) + 1 if hit else 0
            ctx = rc._context(r["context_idx"]) if r["context_idx"] else ""   # texto dos trechos recuperados
            fid = judge_faithfulness(r["answer"], ctx)   # fidelidade: a resposta se ancora nos trechos?
            item[cfg] = {"lat": round(lat, 2), "tok": tok, "hit": hit, "rank": rank, "fidelidade": fid,
                         "answer": r["answer"], "docs": docs}
            ans[cfg] = r["answer"]
            print(f"  {q['id']:<9} {cfg:<9} {lat:5.1f}s {tok:>5}tok hit={hit} fid={fid}", flush=True)
        notas = {cfg: judge_one(q["pergunta"], q["resposta_referencia"], ans[cfg])
                 for cfg in ("baseline", "rag", "proposta")}
        for cfg in ("baseline", "rag", "proposta"):
            item[cfg]["nota"] = notas[cfg]
        print(f"    -> notas {notas}", flush=True)
        rows.append(item)
        _save(path, {"rows": rows})
    return rows

def aggregate(rows):
    if not rows:
        return {}
    out = {}
    for c, _ in CFGS:
        notas = [r[c]["nota"] for r in rows]
        d = {"qualidade": sum(notas) / (2 * len(notas)),
             "pct_correto": sum(n == 2 for n in notas) / len(notas),
             "fidelidade": sum(r[c].get("fidelidade", 0) for r in rows) / (2 * len(rows)),  # 0/1/2 -> 0-1
             "lat_med": sum(r[c]["lat"] for r in rows) / len(rows),
             "tok_med": sum(r[c]["tok"] for r in rows) / len(rows)}
        if c in ("rag", "proposta"):
            d["hit_rate"] = sum(r[c]["hit"] for r in rows) / len(rows)
            d["mrr"] = sum((1.0 / r[c]["rank"]) for r in rows if r[c]["rank"] > 0) / len(rows)
        out[c] = d
    return out

def graficos(agg, n, tag):
    labels = ["Baseline", "RAG", "Proposta"]
    cols = ["#9aa0a6", "#4285F4", "#34A853"]
    order = ("baseline", "rag", "proposta")
    fig, ax = plt.subplots(1, 4, figsize=(17, 4))
    ax[0].bar(labels, [agg[c]["qualidade"] for c in order], color=cols)
    ax[0].set_title("Qualidade da resposta (0-1)\nIA-juiza vs gabarito"); ax[0].set_ylim(0, 1)
    ax[1].bar(labels, [agg[c]["fidelidade"] for c in order], color=cols)
    ax[1].set_title("Fidelidade (0-1)\nresposta ancorada nos trechos"); ax[1].set_ylim(0, 1)
    ax[2].bar(["RAG", "Proposta"], [agg["rag"]["hit_rate"], agg["proposta"]["hit_rate"]], color=cols[1:])
    ax[2].set_title("Acerto da busca (hit-rate)\nfonte certa no contexto"); ax[2].set_ylim(0, 1)
    ax[3].bar(labels, [agg[c]["lat_med"] for c in order], color=cols)
    ax[3].set_title("Latencia media (s)")
    for a in ax:
        for p in a.patches:
            a.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2, p.get_height()),
                       ha="center", va="bottom", fontsize=9)
    fig.suptitle(f"Comparacao Baseline x RAG x Proposta  ({tag}, n={n})", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT, f"comparacao_{tag}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print("Grafico salvo:", path)

if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "all"
    bm = json.load(open(os.path.join(ROOT, "benchmark", "perguntas.json"), encoding="utf-8"))["perguntas"]
    qs = bm if modo == "all" else [q for q in bm if q["id"] in PREVIEW]
    tag = os.environ.get("MODEL_TAG", "")   # ex.: MODEL_TAG=3b / 7b p/ separar os arquivos da ablacao
    path = os.path.join(OUT, f"aval_{modo}{('_' + tag) if tag else ''}.json")
    print(f"Avaliando {len(qs)} perguntas (modo={modo}, modelo={rc.LLM_MODEL})...\n")
    rows = run(qs, path)
    if not rows:
        print("Nenhuma pergunta avaliada."); sys.exit(0)
    agg = aggregate(rows)
    _save(path, {"modo": modo, "n": len(rows), "agg": agg, "rows": rows})
    graficos(agg, len(rows), f"{modo}{('_' + tag) if tag else ''}")
    print("\n=== RESUMO ===")
    for c, _ in CFGS:
        a = agg[c]
        extra = f" | busca: hit={a.get('hit_rate', 0):.0%} MRR={a.get('mrr', 0):.2f}" if "hit_rate" in a else ""
        print(f"{c:<9} qualidade={a['qualidade']:.2f} fidelidade={a['fidelidade']:.2f} "
              f"correto={a['pct_correto']:.0%} lat={a['lat_med']:.1f}s tok={a['tok_med']:.0f}{extra}")

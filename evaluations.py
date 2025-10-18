# evaluations.py
# Single-file evaluator that uses your DB as ground truth.
# - Parser eval: re-parse original files and compare to receipts.db
# - Chat eval: auto-generate a few numeric QA checks from the DB and query Qwen (or your model)

import os, re, json, time, argparse, sqlite3, glob
from statistics import mean
from typing import List, Tuple
import rapidfuzz

# ---------- CONFIG ----------
RECEIPTS_DIR = "receipts"         # where your original PDFs/JPGs are (filenames must match receipts.file_name)
OUT_DIR = "eval/out"              # where summaries will be written
os.makedirs(OUT_DIR, exist_ok=True)

# tolerances / thresholds
PRICE_TOL = 0.50
QTY_TOL   = 0.10
DESC_SIM  = 80

# default chat model (override with --model or env CHAT_MODEL)
DEFAULT_CHAT_MODEL = os.environ.get("CHAT_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

# ---------- project code ----------
from parser import extract_from_pdf, extract_from_image
from utils import read_file_bytes

try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None  # ok, will use fallback matcher


# ---------- small helpers ----------
def _norm(s): return (s or "").strip().lower()

def _desc_sim(a, b):
    if fuzz:
        return fuzz.token_sort_ratio(a or "", b or "")
    a=_norm(a); b=_norm(b)
    if not a or not b: return 0
    A=set(a.split()); B=set(b.split())
    inter=len(A & B); uni=len(A | B)
    return int(100*inter/max(1, uni))

def _first_number(text: str):
    import re
    nums = re.findall(r"-?\d+(?:[.,]\d+)?", text or "")
    if not nums:
        return None
    # return the LAST numeric token in the answer (usually the final result)
    return float(nums[-1].replace(",", "."))

# ====================== PARSER EVAL (vs DB) ======================
def _match_items(pred_items:List[dict], gold_items:List[dict])->Tuple[int,int,int]:
    matched_p=set(); matched_g=set(); tp=0
    for gi, g in enumerate(gold_items):
        best=(-1, None)
        for pi, p in enumerate(pred_items):
            if pi in matched_p: continue
            score=_desc_sim(g.get("description",""), p.get("description",""))
            price_ok = (g.get("line_total") is not None and
                        abs((p.get("line_total") or 0)-(g.get("line_total") or 0)) <= PRICE_TOL)
            qty_ok=True
            if g.get("qty") is not None and p.get("qty") is not None:
                qty_ok = abs((p.get("qty") or 0)-(g.get("qty") or 0)) <= QTY_TOL
            if score >= DESC_SIM and price_ok and qty_ok:
                if score > best[0]:
                    best=(score, pi)
        if best[1] is not None:
            tp += 1
            matched_g.add(gi); matched_p.add(best[1])
    fp = len(pred_items) - len(matched_p)
    fn = len(gold_items) - len(matched_g)
    return tp, fp, fn

def evaluate_parser_from_db(limit:int=30):
    import sqlite3, json, time, glob
    from statistics import mean

    if not os.path.exists("receipts.db"):
        print("receipts.db not found"); return {}

    con = sqlite3.connect("receipts.db"); con.row_factory = sqlite3.Row
    cur = con.cursor()

    # ---- detect receipts schema ----
    cur.execute("PRAGMA table_info(receipts)")
    rcols = {row["name"] for row in cur.fetchall()}

    # primary key
    rid_col = "id" if "id" in rcols else ("receipt_id" if "receipt_id" in rcols else None)
    if not rid_col:
        con.close()
        raise RuntimeError("Could not find receipts primary key (need 'id' or 'receipt_id').")

    file_col  = "file_name" if "file_name" in rcols else ("filename" if "filename" in rcols else None)
    store_col = "store_name" if "store_name" in rcols else ("store" if "store" in rcols else None)
    date_col  = "date" if "date" in rcols else None
    total_col = "total" if "total" in rcols else None
    curr_col  = "currency" if "currency" in rcols else None
    vend_col  = "vendor" if "vendor" in rcols else None

    # build SELECT with aliases
    sel = [f"{rid_col} AS rid"]
    if file_col:  sel.append(f"{file_col} AS file_name")
    if store_col: sel.append(f"{store_col} AS store_name")
    if date_col:  sel.append(f"{date_col} AS date")
    if total_col: sel.append(f"{total_col} AS total")
    if curr_col:  sel.append(f"{curr_col} AS currency")
    if vend_col:  sel.append(f"{vend_col} AS vendor")
    sel_sql = ", ".join(sel)

    cur.execute(f"SELECT {sel_sql} FROM receipts ORDER BY {rid_col} DESC LIMIT ?", (limit,))
    rec_rows = cur.fetchall()

    # ---- detect items schema ----
    cur.execute("PRAGMA table_info(items)")
    icols = {row["name"] for row in cur.fetchall()}
    # expected columns with fallbacks
    desc_col = "description" if "description" in icols else None
    code_col = "item_code" if "item_code" in icols else ("article_number" if "article_number" in icols else None)
    qty_col  = "qty" if "qty" in icols else ("quantity" if "quantity" in icols else None)
    up_col   = "unit_price" if "unit_price" in icols else ("price" if "price" in icols else None)
    lt_col   = "line_total" if "line_total" in icols else ("total" if "total" in icols else None)
    note_col = "promo_note" if "promo_note" in icols else None

    # which foreign key name exists?
    fk_col = "receipt_id" if "receipt_id" in icols else ("rid" if "rid" in icols else None)
    if not fk_col:
        con.close()
        raise RuntimeError("Could not find items→receipts foreign key (need 'receipt_id' or 'rid').")

    # helper to fetch items for a receipt id
    def fetch_items(rid):
        parts = []
        if desc_col: parts.append(f"{desc_col} AS description")
        if code_col: parts.append(f"{code_col} AS item_code")
        if qty_col:  parts.append(f"{qty_col} AS qty")
        if up_col:   parts.append(f"{up_col} AS unit_price")
        if lt_col:   parts.append(f"{lt_col} AS line_total")
        if note_col: parts.append(f"{note_col} AS promo_note")
        sel_items = ", ".join(parts) if parts else "*"
        cur.execute(f"SELECT {sel_items} FROM items WHERE {fk_col} = ?", (rid,))
        return [dict(x) for x in cur.fetchall()]

    # ---- now do the evaluation ----
    per=[]; t_all=[]; field_ok=0; field_tot=0; TP=FP=FN=0

    for r in rec_rows:
        rid   = r["rid"]
        fname = r["file_name"] if "file_name" in r.keys() else None
        gold = {
            "store_name": r["store_name"] if "store_name" in r.keys() else None,
            "date":       r["date"] if "date" in r.keys() else None,
            "total":      r["total"] if "total" in r.keys() else None,
            "currency":   r["currency"] if "currency" in r.keys() else None,
            "vendor":     r["vendor"] if "vendor" in r.keys() else None,
            "line_items": fetch_items(rid)
        }

        # find the source file
        path = None
        if fname:
            p1 = os.path.join(RECEIPTS_DIR, fname)
            if os.path.exists(p1): path = p1
            else:
                g = glob.glob(os.path.join(RECEIPTS_DIR, "**", fname), recursive=True)
                if g: path = g[0]
        if not path:
            print(f"[skip #{rid}] source file not found for {fname}"); continue

        data, name = read_file_bytes(path)
        t0=time.time()
        if name.lower().endswith(".pdf"):
            rec = extract_from_pdf(data, name)
        else:
            rec = extract_from_image(data, name)
        dt=time.time()-t0; t_all.append(dt)
        pred = rec.model_dump()

        # field accuracy (only count fields that exist in DB)
        def add(c): 
            nonlocal field_ok, field_tot
            field_tot += 1; field_ok += 1 if c else 0
        if gold["store_name"] is not None:
            add((_norm(pred.get("store_name")) == _norm(gold["store_name"])))
        if gold["date"] is not None:
            add(pred.get("date") == gold["date"])
        if gold["total"] is not None:
            add(abs((pred.get("total") or 0) - (gold["total"] or 0)) <= PRICE_TOL)
        if gold["currency"] is not None:
            add((_norm(pred.get("currency")) == _norm(gold["currency"])))

        # item matching
        tp, fp, fn = _match_items(pred.get("line_items", []), gold["line_items"])
        TP += tp; FP += fp; FN += fn

        per.append({"receipt_id": rid, "file": fname, "latency_s": round(dt,3), "tp": tp, "fp": fp, "fn": fn})

    con.close()

    prec = TP / max(1, (TP+FP))
    rec  = TP / max(1, (TP+FN))
    f1   = 0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)
    field_acc = field_ok / max(1, field_tot)

    summary = {
        "field_accuracy": round(field_acc,3),
        "item_precision": round(prec,3),
        "item_recall": round(rec,3),
        "item_f1": round(f1,3),
        "latency_mean_s": round(mean(t_all),2) if t_all else None,
        "n_files": len(per)
    }
    os.makedirs("eval/out", exist_ok=True)
    json.dump(per, open(os.path.join("eval/out","parser_per_file.json"),"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(summary, open(os.path.join("eval/out","parser_summary.json"),"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    print("[Parser vs DB] Summary:", summary)
    return summary


# ====================== CHAT EVAL (auto from DB) ======================
SYSTEM_PROMPT = (
    "You are a helpful assistant for Swedish shopping receipts. "
    "Only answer using the receipts context below. Prices are SEK unless stated. "
    "If unsure, say you don't know."
)
RECEIPTS_SCHEMA = (
    "Receipts JSON structure:\n"
    "- A list of receipts. Each receipt object has:\n"
    "  - receipt_id: int\n"
    "  - store_name: string\n"
    "  - date: string (YYYY-MM-DD)\n"
    "  - total: number (SEK)\n"
    "  - currency: string\n"
    "  - vendor: string | null\n"
    "  - items: list of {description, article_number, price, quantity, total}\n"
    "Only use these fields from the RECEIPTS_CONTEXT when answering. Do not invent values."
)

def _build_context_from_db(max_chars=2000) -> str:
    con = sqlite3.connect("receipts.db"); con.row_factory = sqlite3.Row
    cur = con.cursor()

    # receipts + store name (no vendor column in this schema)
    cur.execute("""
        SELECT r.receipt_id AS rid,
               s.name       AS store_name,
               r.date       AS date,
               r.total      AS total
        FROM receipts r
        LEFT JOIN stores s ON r.store_id = s.store_id
        ORDER BY r.receipt_id DESC
        LIMIT 50
    """)
    rec_rows = cur.fetchall()

    recs = []
    for r in rec_rows:
        rid = r["rid"]

        # items mapped to the schema you want in the chat context
        cur.execute("""
            SELECT description,
                   article_number,
                   quantity,
                   price,
                   total
            FROM items
            WHERE receipt_id = ?
            ORDER BY item_id ASC
        """, (rid,))
        items = [dict(x) for x in cur.fetchall()]

        recs.append({
            "receipt_id": rid,
            "store_name": r["store_name"],
            "date":       r["date"],
            "total":      r["total"],
            "currency":   "SEK",             # your DB has no currency; assume SEK
            "vendor":     None,              # no vendor column; keep None
            "items": [{
                "description":    it.get("description"),
                "article_number": it.get("article_number"),
                "price":          it.get("price"),
                "quantity":       it.get("quantity"),
                "total":          it.get("total")
            } for it in items]
        })

    con.close()
    s = json.dumps(recs, ensure_ascii=False)
    return s[:max_chars] if len(s) > max_chars else s


def _ground_truth_from_db():
    con = sqlite3.connect("receipts.db"); con.row_factory = sqlite3.Row
    cur = con.cursor()

    # overall spend
    cur.execute("SELECT SUM(total) AS s FROM receipts")
    overall = float(cur.fetchone()["s"] or 0)

    # average receipt
    cur.execute("SELECT AVG(total) AS a FROM receipts")
    avg = float(cur.fetchone()["a"] or 0)

    # top STORE by spend (join stores because there is no vendor column)
    cur.execute("""
        SELECT s.name AS store_name, SUM(r.total) AS s
        FROM receipts r
        LEFT JOIN stores s ON r.store_id = s.store_id
        GROUP BY s.store_id
        ORDER BY s DESC
        LIMIT 1
    """)
    row = cur.fetchone()
    top_store = row["store_name"] if row else None
    top_sum   = float(row["s"] or 0) if row else 0.0
    con.close()

    qa = [
        {"q": "What is my total spend overall?",        "expect_num": round(overall,2), "tol": 1.0},
        {"q": "What is my average spend per receipt?",  "expect_num": round(avg,2),     "tol": 1.0},
    ]
    if top_store:
        qa.append({"q": f"What is my total spend at {top_store}?", "expect_num": round(top_sum,2), "tol": 1.0})
    return qa



def _load_chat_pipeline(model_id:str):
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype="auto", low_cpu_mem_usage=True
    )
    gen = pipeline(
        "text-generation", model=mdl, tokenizer=tok,
        max_new_tokens=512, do_sample=True, temperature=0.2, top_p=0.9
    )
    return tok, gen

def evaluate_chat_auto(model_id:str=DEFAULT_CHAT_MODEL):
    if not os.path.exists("receipts.db"):
        print("receipts.db not found"); return {}
    try:
        tok, gen = _load_chat_pipeline(model_id)
    except Exception as e:
        print("Could not load model. Install dependencies: pip install transformers accelerate (and torch if needed).")
        raise

    ctx = _build_context_from_db(2000)
    qa = _ground_truth_from_db()
    if not qa:
        print("[Chat] No QA generated from DB."); return {}

    times=[]; correct=0; rows=[]
    for ex in qa:
        messages = [
            {"role":"system","content": SYSTEM_PROMPT + "\n" + RECEIPTS_SCHEMA},
            {"role":"user","content": f"{ctx}\n\nQuestion: {ex['q']}"}
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        t0=time.time()
        out = gen(prompt)[0]["generated_text"]
        dt=time.time()-t0
        ans = out[len(prompt):].strip()
        pred = _first_number(ans)
        ok = (pred is not None and abs(pred - ex["expect_num"]) <= ex["tol"])
        correct += 1 if ok else 0
        times.append(dt)
        rows.append({"q":ex["q"],"ans":ans,"pred_num":pred,"expect_num":ex["expect_num"],"ok":ok,"latency_s":round(dt,2)})
        print(f"Q: {ex['q']}\nA: {ans}\n→ {pred} vs {ex['expect_num']} ok={ok} t={dt:.2f}s\n")

    acc = correct / len(qa)
    summary = {
        "numeric_accuracy": round(acc,3),
        "latency_mean_s": round(mean(times),2) if times else None,
        "n_q": len(qa),
        "model": model_id
    }
    json.dump(rows, open(os.path.join(OUT_DIR,"chat_per_q.json"),"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(summary, open(os.path.join(OUT_DIR,"chat_summary.json"),"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    print("[Chat auto] Summary:", summary)
    return summary


# ====================== CLI ======================
def main():
    ap = argparse.ArgumentParser(description="Evaluate parser and chat using DB as ground truth.")
    ap.add_argument("--parser", action="store_true", help="run parser eval (re-parse files vs DB)")
    ap.add_argument("--chat",   action="store_true", help="run chat eval (auto QA from DB)")
    ap.add_argument("--all",    action="store_true", help="run both")
    ap.add_argument("--limit",  type=int, default=30, help="how many latest receipts to evaluate")
    ap.add_argument("--model",  type=str, default=DEFAULT_CHAT_MODEL, help="HF model id for chat (default Qwen2.5-1.5B)")
    args = ap.parse_args()

    ran=False
    if args.all or args.parser:
        evaluate_parser_from_db(limit=args.limit); ran=True
    if args.all or args.chat:
        evaluate_chat_auto(model_id=args.model); ran=True
    if not ran:
        ap.print_help()

if __name__ == "__main__":
    main()

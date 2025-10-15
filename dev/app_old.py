import io, re, uuid, json, warnings, os, traceback
from datetime import datetime
from typing import List, Optional, Dict

import numpy as np
from PIL import Image

import gradio as gr
import fitz  # PyMuPDF
import pdfplumber
import easyocr

from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Import from database.py
from database import (
    create_database, add_store, add_item, get_all_receipts_from_db,
    get_database_stats, get_recent_items, clear_database, search_items
)

# Data models
class LineItem(BaseModel):
    item_code: Optional[str] = None
    description: str
    qty: float = 1.0
    unit_price: Optional[float] = None
    line_total: Optional[float] = None
    # discount attachment
    is_discounted: bool = False
    discount_amount_total: float | None = None
    unit_price_after_discount: float | None = None
    line_total_after_discount: float | None = None
    promo_note: Optional[str] = None  # e.g., "Läsk 3f50 kr"

class Receipt(BaseModel):
    doc_id: str
    source_type: str          # "pdf" | "image"
    file_name: str
    vendor: Optional[str] = None
    date: Optional[str] = None
    currency: Optional[str] = None
    total: Optional[float] = None
    line_items: List[LineItem] = Field(default_factory=list)
    preview: Optional[str] = None
    store_id: Optional[int] = None  # Added for DB reference

RECEIPTS: Dict[str, Receipt] = {}

# Debug - ignore warnings
warnings.filterwarnings("ignore")

# ---------------- Config ----------------
CHAT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"   
SYSTEM_PROMPT = (
    "You are a helpful assistant for Swedish shopping receipts. "
    "Only answer using the receipts context below. Prices are SEK unless stated. "
    "If unsure, say you don't know."
)

OCR_LANGS = ['sv','en'] # Swedish + English
MAX_CTX_JSON_CHARS = 6000
PDF_MAX_PAGES = 4
IMG_MAX = (1600, 1600)
EXCLUDE_PANT = True
INCLUDE_DISCOUNTS = True

# Init OCR (uses CUDA if available)
reader = easyocr.Reader(OCR_LANGS, gpu=True)

# Init chat model (GPU if available)
tok = AutoTokenizer.from_pretrained(CHAT_MODEL)
mdl = AutoModelForCausalLM.from_pretrained(CHAT_MODEL)
chat = pipeline("text-generation", model=mdl, tokenizer=tok, max_new_tokens=256, temperature=0.2, top_p=0.9)

# ---------------- Regex / helpers ----------------
CURRENCY_RE = re.compile(r"(SEK|kr|kronor)", re.I)
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2}|\d{2}[./-]\d{2}[./-]\d{2,4})")
TOTAL_RE = re.compile(r"\b(total|summa|betalat|att\s+betala)\b[:\s]*([\d.,]+)", re.I)

# headers / non-product rows to ignore
HEADER_RE = re.compile(r"^\s*(Beskrivning|Artikeln(ummer|r)?|Pris|Mängd|Summa\s*\(SEK\))\s*$", re.I)
SKIP_ROW_RE = re.compile(
    r"\b(köp|varav\s+moms|moms|netto|brutto|total(t)?\s*(sek)?|att\s+betala|betalat|kort|erhållen\s+rabatt|avrundning)\b",
    re.I,
)
# Pant rows
PANT_RE = re.compile(r"^\s*\*?\s*pant\b", re.I)

# Discount hints & bundle patterns (e.g., "Läsk 3f50 kr")
DISCOUNT_HINT_RE = re.compile(r"\b(rabatt|kampanj|extrapris|prisjustering|kupong)\b", re.I)
BUNDLE_RE = re.compile(r"(?P<n>\d+)\s*f\s*(?P<price>\d+(?:[.,]\d{2})?)", re.I)  # "3f50", "3F50", "3f50 kr"

# ICA-style table row:
# Desc   ArticleNumber   UnitPrice   Qty [unit]   LineTotal
TABLE_ROW_RE = re.compile(
    r"""^
        (?P<desc>.+?)\s+
        (?P<code>\d{6,14})\s+
        (?P<unit>-?\d+(?:[.,]\d{2}))\s+
        (?P<qty>\d+(?:[.,]\d{1,3})?)      # 1 or 1,00 or 3
        (?:\s*(?:st|kg|g|l|L))?           # optional unit token
        \s+
        (?P<sum>-?\d+(?:[.,]\d{2}))
        \s*$
    """,
    re.X
)

# Fallback: "desc ... sum" (no code/qty known) – require at least one letter in desc to avoid numeric-only lines
PRICE_AT_END_FALLBACK_RE = re.compile(
    r"""^
        (?P<desc>(?=.*[A-Za-zÅÄÖåäö]).{3,80}?)\s+
        (?P<sum>-?\d+(?:[.,]\d{2}))
        \s*(?:kr|SEK)?
        \s*$
    """,
    re.X
)

def read_file_bytes(f):
    if hasattr(f, "read"):
        try:
            data = f.read()
            name = os.path.basename(getattr(f, "name", "uploaded"))
            return data, name
        except Exception:
            pass
    path = getattr(f, "path", None)
    if path and os.path.exists(path):
        with open(path, "rb") as fp:
            data = fp.read()
        return data, os.path.basename(path)
    if isinstance(f, dict) and "path" in f and os.path.exists(f["path"]):
        with open(f["path"], "rb") as fp:
            data = fp.read()
        return data, os.path.basename(f["path"])
    if isinstance(f, (str, os.PathLike)) and os.path.exists(str(f)):
        p = str(f)
        with open(p, "rb") as fp:
            data = fp.read()
        return data, os.path.basename(p)
    raise TypeError(f"Unsupported upload type: {type(f)}; value={repr(f)}")

def to_float(s: Optional[str]) -> Optional[float]:
    if not s: return None
    s = s.replace(" ", "").replace("kr", "").replace("KR", "")
    if s.count(",")==1 and s.count(".")==0:
        s = s.replace(",", ".")
    elif s.count(",")>1 and s.count(".")==0:
        s = s.replace(",", "")
    return float(re.sub(r"[^\d.-]", "", s)) if re.search(r"\d", s) else None

def normalize_date(s: Optional[str]) -> Optional[str]:
    if not s: return None
    s2 = s.replace(".", "-").replace("/", "-")
    for fmt in ("%Y-%m-%d","%d-%m-%Y","%d-%m-%y"):
        try:
            return datetime.strptime(s2, fmt).date().isoformat()
        except:
            pass
    m = DATE_RE.search(s2)
    if m:
        return normalize_date(m.group(1))
    return None

# ---------------- Parsing logic ----------------
def parse_lines_to_receipt(lines: List[str], file_name: str, source_type: str) -> Receipt:
    doc_id = str(uuid.uuid4())
    # vendor guess
    vendor = None
    for L in lines[:30]:
        m = re.search(r"\b(ICA|Willys|Hemköp|Coop|Lidl)\b", L, re.I)
        if m:
            vendor = m.group(1); break

    # date
    date = None
    for L in lines[:80]:
        dm = DATE_RE.search(L)
        if dm:
            d = normalize_date(dm.group(1))
            if d: date = d; break

    # total
    total = None
    for L in lines:
        tm = TOTAL_RE.search(L)
        if tm:
            total = to_float(tm.group(2))
            if total is not None: break

    currency = "SEK" if any(CURRENCY_RE.search(L) for L in lines) else None

    # ---- items (table-first, then fallback + discount attach) ----
    items: List[LineItem] = []
    for L in lines:
        Ls = L.strip()
        if (not Ls) or HEADER_RE.match(Ls) or SKIP_ROW_RE.search(Ls):
            continue
        if EXCLUDE_PANT and PANT_RE.match(Ls):
            continue

        # Table row with article code
        m = TABLE_ROW_RE.match(Ls)
        if m:
            desc = m.group("desc").strip()
            starred = desc.startswith("*")
            if starred: desc = desc.lstrip("*").strip()

            code = m.group("code").strip()
            unit = to_float(m.group("unit"))
            qty  = to_float(m.group("qty")) or 1.0
            lsum = to_float(m.group("sum"))

            if EXCLUDE_PANT and PANT_RE.match(desc):
                continue

            items.append(LineItem(
                item_code=code,
                description=desc,
                qty=qty,
                unit_price=unit,
                line_total=lsum,
            ))
            if starred:
                items[-1].promo_note = "*"   # mark as promo-eligible until we see discount text
            continue

        # Fallback: description ... sum (letters required in desc)
        m2 = PRICE_AT_END_FALLBACK_RE.match(Ls)
        if m2 and not SKIP_ROW_RE.search(Ls):
            desc = m2.group("desc").strip()
            lsum = to_float(m2.group("sum"))

            if EXCLUDE_PANT and PANT_RE.match(desc):
                continue

            # Negative amount? attach to the last product as discount
            if INCLUDE_DISCOUNTS and lsum is not None and lsum < 0 and items:
                # attach to the latest product-like row
                target_idx = None
                for idx in range(len(items)-1, -1, -1):
                    it = items[idx]
                    if it.item_code or it.unit_price is not None or it.line_total is not None:
                        target_idx = idx; break
                if target_idx is not None:
                    it = items[target_idx]
                    prev_disc = it.discount_amount_total or 0.0
                    disc_total = prev_disc + lsum  # lsum negative
                    it.discount_amount_total = disc_total
                    it.is_discounted = True

                    # Try to parse explicit bundle like "3f50"
                    bundle_match = BUNDLE_RE.search(desc)
                    if bundle_match:
                        n = int(bundle_match.group("n"))
                        bundle_price = to_float(bundle_match.group("price"))
                        if n and bundle_price is not None:
                            it.line_total_after_discount = round(bundle_price, 2)
                            it.unit_price_after_discount = round(bundle_price / n, 2)
                        else:
                            # fallback to arithmetic
                            base_total = (it.line_total if it.line_total is not None
                                          else (it.unit_price or 0.0) * (it.qty or 1.0))
                            after = base_total + lsum
                            it.line_total_after_discount = round(after, 2)
                            if it.qty:
                                it.unit_price_after_discount = round(after / it.qty, 2)
                    else:
                        # no bundle text → compute from base total
                        base_total = (it.line_total if it.line_total is not None
                                      else (it.unit_price or 0.0) * (it.qty or 1.0))
                        after = base_total + lsum
                        it.line_total_after_discount = round(after, 2)
                        if it.qty:
                            it.unit_price_after_discount = round(after / it.qty, 2)

                    # replace "*" marker with the actual promo text
                    it.promo_note = desc if it.promo_note in (None, "*") else (it.promo_note + " | " + desc)
                    continue  # don't append as its own item

    # Clean-up: if something was marked "*" but got no discount line, drop the marker
    for it in items:
        if it.promo_note == "*":
            it.promo_note = None

    return Receipt(
        doc_id=doc_id,
        source_type=source_type,
        file_name=file_name,
        vendor=vendor,
        date=date,
        currency=currency,
        total=total,
        line_items=items,
        preview="\n".join(lines[:80])
    )

# ---- PDF: text extraction; if page has no text, rasterize -> OCR ----
def pix_to_text(page) -> List[str]:
    mat = fitz.Matrix(2, 2)  # ~150–200 dpi
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    img.thumbnail(IMG_MAX)
    arr = np.array(img)
    res = reader.readtext(arr, detail=1, paragraph=True)
    text = "\n".join([r[1] for r in res]) if res else ""
    return [L.strip() for L in text.splitlines() if L.strip()]

def pdf_to_lines(pdf_bytes: bytes, pages=PDF_MAX_PAGES) -> List[str]:
    lines: List[str] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        n = min(pages, len(doc))
        for i in range(n):
            page = doc[i]
            txt = page.get_text("text") or ""
            if txt.strip():
                for L in txt.splitlines():
                    Ls = L.strip()
                    if Ls:
                        lines.append(Ls)
            else:
                lines.extend(pix_to_text(page))
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for p in pdf.pages[:pages]:
                t = p.extract_text() or ""
                for L in t.splitlines():
                    Ls = L.strip()
                    if Ls and Ls not in lines:
                        lines.append(Ls)
    except Exception:
        pass
    return lines

def extract_from_pdf(pdf_bytes: bytes, file_name: str) -> Receipt:
    lines = pdf_to_lines(pdf_bytes, pages=PDF_MAX_PAGES)
    return parse_lines_to_receipt(lines, file_name, "pdf")

# ---------------- Save receipt to database ----------------
def save_receipt_to_db(receipt: Receipt):
    """Save a receipt to the database."""
    # Add store (or get existing)
    store_id, _ = add_store(
        name=receipt.vendor or "Unknown Store",
        org_number=None  # We don't extract org_number from receipts yet
    )
    receipt.store_id = store_id
    
    # Add each item
    purchase_date = receipt.date or datetime.now().date().isoformat()
    
    stats = {"new": 0, "existing": 0}
    for item in receipt.line_items:
        # Calculate final price and discount
        final_total = item.line_total_after_discount if item.is_discounted else item.line_total
        discount_amount = abs(item.discount_amount_total) if item.discount_amount_total else 0.0
        
        _, was_created = add_item(
            description=item.description,
            article_number=item.item_code,
            price=item.unit_price,
            quantity=item.qty,
            total=final_total,
            discount=discount_amount,
            category=None,  # We don't extract category yet
            store_id=store_id,
            purchase_date=purchase_date,
            comparison_price=None,
            comparison_price_unit=None
        )
        
        if was_created:
            stats["new"] += 1
        else:
            stats["existing"] += 1
    
    return stats

# ---------------- Chat ----------------
def build_context() -> str:
    """Build context from both in-memory receipts and database."""
    rows = []
    
    # Add in-memory receipts
    for r in RECEIPTS.values():
        rows.append({
            "doc_id": r.doc_id,
            "vendor": r.vendor,
            "date": r.date,
            "currency": r.currency,
            "total": r.total,
            "items": [
                {
                    "code": li.item_code,
                    "desc": li.description,
                    "qty": li.qty,
                    "unit_price": li.unit_price,
                    "line_total": li.line_total,
                    "is_discounted": li.is_discounted,
                    "discount_amount_total": li.discount_amount_total,
                    "unit_price_after_discount": li.unit_price_after_discount,
                    "line_total_after_discount": li.line_total_after_discount,
                    "promo_note": li.promo_note
                } for li in r.line_items
            ]
        })
    
    # Add database receipts
    db_receipts = get_all_receipts_from_db()
    for r in db_receipts:
        rows.append({
            "vendor": r["vendor"],
            "date": r["date"],
            "items": r["items"]
        })
    
    s = json.dumps(rows, ensure_ascii=False)
    return s[:MAX_CTX_JSON_CHARS]


def chat_respond(message, history):
    if not RECEIPTS and not get_all_receipts_from_db():
        reply = "I have no receipts yet. Please upload a PDF or image first."
        return history + [(message, reply)], gr.update(value="")
    ctx = build_context()
    transcript = ""
    for u, a in history:
        transcript += f"User: {u}\nAssistant: {a}\n"
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"RECEIPTS_CONTEXT:\n{ctx}\n\n"
        f"{transcript}"
        f"User: {message}\nAssistant:"
    )
    out = chat(prompt)[0]["generated_text"]
    reply = out.split("Assistant:")[-1].strip()
    return history + [(message, reply)], gr.update(value="")

# ---------------- Database view helpers ----------------
def view_database():
    """Return database contents as formatted text."""
    stats = get_database_stats()
    items = get_recent_items(50)
    
    output = f"**Database Statistics**\n"
    output += f"- Total Stores: {stats['store_count']}\n"
    output += f"- Total Items: {stats['item_count']}\n"
    if stats['earliest_date']:
        output += f"- Date Range: {stats['earliest_date']} to {stats['latest_date']}\n"
    output += "\n**Recent Items (last 50):**\n\n"
    
    for item in items:
        store, date, desc, price, qty, total, discount = item
        discount_str = f" (discount: {discount} kr)" if discount else ""
        output += f"- **{store}** ({date}): {desc} - {qty}x @ {price} kr = {total} kr{discount_str}\n"
    
    return output

def clear_db_wrapper():
    """Wrapper to clear database and in-memory receipts."""
    clear_database()
    RECEIPTS.clear()
    return "Database cleared successfully!"

# ---------------- Gradio UI ----------------
def do_extract(files):
    results = []
    for f in files:
        try:
            data, name = read_file_bytes(f)
            if name.lower().endswith(".pdf"):
                rec = extract_from_pdf(data, name)
            # else:
                # rec = extract_from_image(data, name)
            
            # Save to database
            stats = save_receipt_to_db(rec)
            
            # Keep in memory too
            RECEIPTS[rec.doc_id] = rec
            
            result = rec.model_dump()
            result["db_stats"] = stats
            results.append(result)
        except Exception as e:
            print("[extract error]", name if 'name' in locals() else f, "->", e)
            traceback.print_exc()
            results.append({"file": name if 'name' in locals() else str(f), "error": str(e)})
    return results

with gr.Blocks(title="Receipt Analyzer (Conda + DB)") as demo:
    gr.Markdown("## 🧾 Receipt Analyzer\nUpload PDFs or images → extract → auto-save to database → chat about prices")
    
    with gr.Tab("1) Upload & Extract"):
        files = gr.File(file_count="multiple", file_types=[".pdf", ".png", ".jpg", ".jpeg"], label="Upload receipts")
        out = gr.JSON(label="Parsed receipts (JSON) - includes DB stats")
        gr.Button("Extract & Save to DB").click(do_extract, inputs=files, outputs=out)

    with gr.Tab("2) Chat"):
        chatbot = gr.Chatbot(label="Chat", height=420)
        msg = gr.Textbox(label="Ask (press Enter to send)", placeholder="e.g., What did milk cost at ICA in June?", lines=1)
        send = gr.Button("Send", variant="primary")
        clear = gr.Button("Clear")
        send.click(chat_respond, inputs=[msg, chatbot], outputs=[chatbot, msg])
        msg.submit(chat_respond, inputs=[msg, chatbot], outputs=[chatbot, msg])
        clear.click(lambda: ([], ""), outputs=[chatbot, msg])
    
    with gr.Tab("3) Database View"):
        gr.Markdown("### View stored receipts and items")
        db_output = gr.Markdown()
        with gr.Row():
            refresh_btn = gr.Button("Refresh View", variant="primary")
            clear_db_btn = gr.Button("Clear Database", variant="stop")
        
        refresh_btn.click(view_database, outputs=db_output)
        clear_db_btn.click(clear_db_wrapper, outputs=db_output)
        
        # Load database view on tab open
        demo.load(view_database, outputs=db_output)

if __name__ == "__main__":
    # Initialize database on startup
    print("[startup] Creating database if needed...")
    create_database()
    print("[run] starting Gradio on http://127.0.0.1:7860 …")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
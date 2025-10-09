# app.py
import io, re, uuid, json, warnings
from datetime import datetime
from typing import List, Optional, Dict
import os, traceback
import numpy as np
from PIL import Image

import gradio as gr
import fitz  # PyMuPDF
import pdfplumber
import easyocr

from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

warnings.filterwarnings("ignore")

# ---------------- Config ----------------
CHAT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"   # small & quick to test
OCR_LANGS = ['sv','en']                              # Swedish + English
MAX_CTX_JSON_CHARS = 6000
PDF_MAX_PAGES = 4
IMG_MAX = (1600, 1600)

# Init OCR (uses CUDA if available)
reader = easyocr.Reader(OCR_LANGS, gpu=True)

# Init chat model (GPU if available)
tok = AutoTokenizer.from_pretrained(CHAT_MODEL)
mdl = AutoModelForCausalLM.from_pretrained(CHAT_MODEL)
chat = pipeline("text-generation", model=mdl, tokenizer=tok, max_new_tokens=256, temperature=0.2, top_p=0.9)

# ---------------- Regex / helpers ----------------
CURRENCY_RE = re.compile(r"(SEK|kr|kronor)", re.I)
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2}|\d{2}[./-]\d{2}[./-]\d{2,4})")
TOTAL_RE = re.compile(r"\b(total|summa)\b[:\s]*([\d.,]+)", re.I)
PRICE_LINE_RE = re.compile(r"(.{3,60}?)\s+(\d+(?:[.,]\d{2}))\s*(kr|SEK)?$", re.I)
ITEM_WITH_QTY_RE = re.compile(r"(.{3,40}?)\s+(\d+(?:[.,]\d{1,2})?)\s*x\s*(\d+(?:[.,]\d{2}))", re.I)

def read_file_bytes(f):
    """
    Robustly read bytes from a Gradio file input.
    Handles: file-like objects, UploadedFile (with .path), str/Path, dict {"path": ...}.
    Returns: (bytes, file_name)
    """
    # file-like (has .read)
    if hasattr(f, "read"):
        try:
            data = f.read()
            name = os.path.basename(getattr(f, "name", "uploaded"))
            return data, name
        except Exception:
            pass

    # Gradio UploadedFile has .path
    path = getattr(f, "path", None)
    if path and os.path.exists(path):
        with open(path, "rb") as fp:
            data = fp.read()
        name = os.path.basename(path)
        return data, name

    # plain path or dict
    if isinstance(f, dict) and "path" in f and os.path.exists(f["path"]):
        with open(f["path"], "rb") as fp:
            data = fp.read()
        name = os.path.basename(f["path"])
        return data, name

    if isinstance(f, (str, os.PathLike)) and os.path.exists(str(f)):
        p = str(f)
        with open(p, "rb") as fp:
            data = fp.read()
        name = os.path.basename(p)
        return data, name

    raise TypeError(f"Unsupported upload type: {type(f)}; value={repr(f)}")

def to_float(s: Optional[str]) -> Optional[float]:
    if not s: return None
    s = s.replace(" ", "").replace("kr", "").replace("KR", "")
    if s.count(",")==1 and s.count(".")==0:
        s = s.replace(",", ".")
    elif s.count(",")>1 and s.count(".")==0:
        s = s.replace(",", "")
    return float(re.sub(r"[^\d.]", "", s)) if re.search(r"\d", s) else None

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

# ---------------- Data models ----------------
class LineItem(BaseModel):
    description: str
    qty: float = 1.0
    unit_price: Optional[float] = None
    line_total: Optional[float] = None

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

RECEIPTS: Dict[str, Receipt] = {}

# ---------------- Parsing logic ----------------
def parse_lines_to_receipt(lines: List[str], file_name: str, source_type: str) -> Receipt:
    doc_id = str(uuid.uuid4())
    # vendor guess (common Swedish chains)
    vendor = None
    for L in lines[:30]:
        m = re.search(r"\b(ICA|Willys|Hemköp|Coop|Lidl)\b", L, re.I)
        if m:
            vendor = m.group(1)
            break

    # date
    date = None
    for L in lines[:80]:
        dm = DATE_RE.search(L)
        if dm:
            d = normalize_date(dm.group(1))
            if d:
                date = d
                break

    # total
    total = None
    for L in lines:
        tm = TOTAL_RE.search(L)
        if tm:
            total = to_float(tm.group(2))
            if total is not None:
                break

    # currency
    currency = "SEK" if any(CURRENCY_RE.search(L) for L in lines) else None

    # line items
    items: List[LineItem] = []
    for L in lines:
        m1 = ITEM_WITH_QTY_RE.search(L)
        m2 = PRICE_LINE_RE.search(L)
        if m1:
            desc = m1.group(1).strip()
            qty = to_float(m1.group(2)) or 1.0
            up = to_float(m1.group(3))
            if not re.search(r"\b(total|summa)\b", L, re.I):
                items.append(LineItem(description=desc, qty=qty, unit_price=up))
        elif m2:
            desc = m2.group(1).strip()
            up = to_float(m2.group(2))
            if not re.search(r"\b(total|summa)\b", L, re.I):
                items.append(LineItem(description=desc, qty=1.0, unit_price=up))
        if len(items) >= 40:
            break

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
    """Rasterize a page and OCR with EasyOCR; return lines."""
    # 150dpi-ish rendering for speed/accuracy tradeoff
    mat = fitz.Matrix(2, 2)  # 2x zoom
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
                # PyMuPDF blocks keep some order; we’ll split lines directly
                for L in txt.splitlines():
                    Ls = L.strip()
                    if Ls:
                        lines.append(Ls)
            else:
                # image-only page → OCR
                ocr_lines = pix_to_text(page)
                lines.extend(ocr_lines)
    # pdfplumber pass can sometimes add missed lines
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

# ---- Images: OCR ----
def extract_from_image(image_bytes: bytes, file_name: str) -> Receipt:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail(IMG_MAX)
    arr = np.array(img)
    res = reader.readtext(arr, detail=1, paragraph=True)
    text = "\n".join([r[1] for r in res]) if res else ""
    lines = [L.strip() for L in text.splitlines() if L.strip()]
    return parse_lines_to_receipt(lines, file_name, "image")

# ---------------- Chat ----------------
def build_context() -> str:
    rows = []
    for r in RECEIPTS.values():
        rows.append({
            "doc_id": r.doc_id,
            "vendor": r.vendor,
            "date": r.date,
            "currency": r.currency,
            "total": r.total,
            "items": [{"desc": li.description, "qty": li.qty, "unit_price": li.unit_price} for li in r.line_items]
        })
    s = json.dumps(rows, ensure_ascii=False)
    return s[:MAX_CTX_JSON_CHARS]

def do_chat(q: str) -> str:
    if not RECEIPTS:
        return "No receipts yet. Upload a PDF or image first."
    ctx = build_context()
    prompt = (
        "You are a helpful assistant for Swedish shopping receipts. "
        "Only answer using the receipts context below. Prices are SEK unless stated. "
        "If unsure, say you don't know.\n\n"
        f"RECEIPTS_CONTEXT:\n{ctx}\n\n"
        f"Question: {q}\nAnswer with doc_id and date when citing prices.\nAssistant:"
    )
    out = chat(prompt)[0]["generated_text"]
    return out.split("Assistant:")[-1].strip()

# ---------------- Gradio UI ----------------
def do_extract(files):
    results = []
    for f in files:
        try:
            data, name = read_file_bytes(f)
            if name.lower().endswith(".pdf"):
                rec = extract_from_pdf(data, name)
            else:
                rec = extract_from_image(data, name)
            RECEIPTS[rec.doc_id] = rec
            results.append(rec.model_dump())
        except Exception as e:
            # print full traceback to the terminal so we can see the real cause
            print("[extract error]", name if 'name' in locals() else f, "->", e)
            traceback.print_exc()
            results.append({"file": name if 'name' in locals() else str(f), "error": str(e)})
    return results

with gr.Blocks(title="Receipt Analyzer (Conda)") as demo:
    gr.Markdown("## 🧾 Receipt Analyzer\nUpload PDFs or images → extract → ask questions (e.g., *What did milk cost at ICA?*)")
    with gr.Tab("1) Upload & Extract"):
        files = gr.File(file_count="multiple", file_types=[".pdf", ".png", ".jpg", ".jpeg"], label="Upload receipts")
        out = gr.JSON(label="Parsed receipts (JSON)")
        gr.Button("Extract").click(do_extract, inputs=files, outputs=out)

    with gr.Tab("2) Chat"):
        q = gr.Textbox(label="Ask (e.g., 'What did milk cost at ICA in June?')")
        a = gr.Textbox(label="Answer")
        gr.Button("Ask").click(do_chat, inputs=q, outputs=a)

if __name__ == "__main__":
    demo.launch()

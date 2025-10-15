# parser.py

import platform
import io
import re
import uuid
from datetime import datetime
from typing import List, Optional

import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import pdfplumber
import easyocr

from pydantic import BaseModel, Field

from utils import (
    to_float, normalize_date
)


# ---------------- Config ----------------
OCR_LANGS = ['sv', 'en']  # Swedish + English
PDF_MAX_PAGES = 4
IMG_MAX = (1600, 1600)
EXCLUDE_PANT = True
INCLUDE_DISCOUNTS = True

# Init OCR - use CPU on macOS for stability
USE_GPU = platform.system() != 'Darwin'  # GPU on Linux/Windows, CPU on Mac
reader = easyocr.Reader(OCR_LANGS, gpu=True)


# ---------------- Data models ----------------
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
    receipt_id: str
    source_type: str  # "pdf" | "image"
    file_name: str
    vendor: Optional[str] = None
    store_name: Optional[str] = None
    address: Optional[str] = None
    post_address: Optional[str] = None
    date: Optional[str] = None
    currency: Optional[str] = None
    total: Optional[float] = None
    line_items: List[LineItem] = Field(default_factory=list)
    preview: Optional[str] = None
    store_id: Optional[int] = None  # Added later DB reference


# ---------------- Regex patterns ----------------
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
BUNDLE_RE = re.compile(r"(?P<n>\d+)\s*f\s*(?P<price>\d+(?:[.,]\d{2})?)", re.I)

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

# Fallback: "desc ... sum" (no code/qty known)
PRICE_AT_END_FALLBACK_RE = re.compile(
    r"""^
        (?P<desc>(?=.*[A-Za-zÅÄÖåäö]).{3,80}?)\s+
        (?P<sum>-?\d+(?:[.,]\d{2}))
        \s*(?:kr|SEK)?
        \s*$
    """,
    re.X
)


# ---------------- Parsing logic ----------------
def parse_lines_to_receipt(lines: List[str], file_name: str, source_type: str) -> Receipt:
    """Parse text lines into a structured Receipt object."""
    receipt_id = str(uuid.uuid4())
    
    print(lines[1])
    
    # Extract vendor, store, address, postaladdress
    vendor = None
    store = None
    i = 0
    for L in lines[:30]:
        m = re.search(r"\b(ICA|Willys|Hemköp|Coop|Lidl)\b", L, re.I)
        if m:
            vendor = m.group(1)
            store_name = L
            if i + 1 < len(lines):
                address = lines[i + 1]
            if i + 2 < len(lines):
                post_address = lines[i + 2]    
            break
        i = i + 1

    # Extract date
    date = None
    for L in lines[:80]:
        dm = DATE_RE.search(L)
        if dm:
            d = normalize_date(dm.group(1))
            if d:
                date = d
                break

    # Extract total
    total = None
    for L in lines:
        tm = TOTAL_RE.search(L)
        if tm:
            total = to_float(tm.group(2))
            if total is not None:
                break

    # Extract currency
    currency = "SEK" if any(CURRENCY_RE.search(L) for L in lines) else None

    # Parse line items
    items: List[LineItem] = []
    for L in lines:
        Ls = L.strip()
        if (not Ls) or HEADER_RE.match(Ls) or SKIP_ROW_RE.search(Ls):
            continue
        if EXCLUDE_PANT and PANT_RE.match(Ls):
            continue

        # Try table row format with article code
        m = TABLE_ROW_RE.match(Ls)
        if m:
            desc = m.group("desc").strip()
            starred = desc.startswith("*")
            if starred:
                desc = desc.lstrip("*").strip()

            code = m.group("code").strip()
            unit = to_float(m.group("unit"))
            qty = to_float(m.group("qty")) or 1.0
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
                items[-1].promo_note = "*"
            continue

        # Try fallback format: description ... sum
        m2 = PRICE_AT_END_FALLBACK_RE.match(Ls)
        if m2 and not SKIP_ROW_RE.search(Ls):
            desc = m2.group("desc").strip()
            lsum = to_float(m2.group("sum"))

            if EXCLUDE_PANT and PANT_RE.match(desc):
                continue

            # Negative amount? Attach as discount to last product
            if INCLUDE_DISCOUNTS and lsum is not None and lsum < 0 and items:
                target_idx = None
                for idx in range(len(items) - 1, -1, -1):
                    it = items[idx]
                    if it.item_code or it.unit_price is not None or it.line_total is not None:
                        target_idx = idx
                        break
                
                if target_idx is not None:
                    it = items[target_idx]
                    prev_disc = it.discount_amount_total or 0.0
                    disc_total = prev_disc + lsum  # lsum is negative
                    it.discount_amount_total = disc_total
                    it.is_discounted = True

                    # Try to parse bundle pricing (e.g., "3f50")
                    bundle_match = BUNDLE_RE.search(desc)
                    if bundle_match:
                        n = int(bundle_match.group("n"))
                        bundle_price = to_float(bundle_match.group("price"))
                        if n and bundle_price is not None:
                            it.line_total_after_discount = round(bundle_price, 2)
                            it.unit_price_after_discount = round(bundle_price / n, 2)
                        else:
                            base_total = (it.line_total if it.line_total is not None
                                        else (it.unit_price or 0.0) * (it.qty or 1.0))
                            after = base_total + lsum
                            it.line_total_after_discount = round(after, 2)
                            if it.qty:
                                it.unit_price_after_discount = round(after / it.qty, 2)
                    else:
                        # No bundle text → compute from base total
                        base_total = (it.line_total if it.line_total is not None
                                    else (it.unit_price or 0.0) * (it.qty or 1.0))
                        after = base_total + lsum
                        it.line_total_after_discount = round(after, 2)
                        if it.qty:
                            it.unit_price_after_discount = round(after / it.qty, 2)

                    # Update promo note
                    it.promo_note = desc if it.promo_note in (None, "*") else (it.promo_note + " | " + desc)
                    continue  # Don't append as its own item

    # Clean-up: remove temporary "*" markers that didn't get discount lines
    for it in items:
        if it.promo_note == "*":
            it.promo_note = None

    return Receipt(
        receipt_id=receipt_id,
        source_type=source_type,
        file_name=file_name,
        vendor=vendor,
        store_name=store_name,
        address=address,
        post_address=post_address,
        date=date,
        currency=currency,
        total=total,
        line_items=items,
        preview="\n".join(lines[:80])
    )


# ---------------- PDF extraction ----------------
def pix_to_text(page) -> List[str]:
    """Convert PDF page to text using OCR."""
    mat = fitz.Matrix(2, 2)  # ~150–200 dpi
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    img.thumbnail(IMG_MAX)
    arr = np.array(img)
    res = reader.readtext(arr, detail=1, paragraph=True)
    text = "\n".join([r[1] for r in res]) if res else ""
    return [L.strip() for L in text.splitlines() if L.strip()]


def pdf_to_lines(pdf_bytes: bytes, pages: int = PDF_MAX_PAGES) -> List[str]:
    """Extract text lines from PDF using multiple methods."""
    lines: List[str] = []
    
    # Method 1: PyMuPDF
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
                # No text found, try OCR
                lines.extend(pix_to_text(page))
    
    # Method 2: pdfplumber (fallback/supplement)
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
    """Extract receipt data from PDF file."""
    lines = pdf_to_lines(pdf_bytes, pages=PDF_MAX_PAGES)
    return parse_lines_to_receipt(lines, file_name, "pdf")


# ---------------- Image extraction ----------------
def extract_from_image(image_bytes: bytes, file_name: str) -> Receipt:
    """Extract receipt data from image file using OCR."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail(IMG_MAX)
    arr = np.array(img)
    res = reader.readtext(arr, detail=1, paragraph=True)
    text = "\n".join([r[1] for r in res]) if res else ""
    lines = [L.strip() for L in text.splitlines() if L.strip()]
    return parse_lines_to_receipt(lines, file_name, "image")
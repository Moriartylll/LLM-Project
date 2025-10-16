# app.py

import json
import traceback
from datetime import datetime
from typing import Dict

import gradio as gr

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Import from database.py
from db import (
    init_db, add_store_to_db, add_receipt_to_db, add_item_to_db, get_all_receipts_from_db,
    get_stats_from_db, get_recent_items, clear_db, search_items, view_db
)

# Import from parser.py
from parser import Receipt, extract_from_pdf, extract_from_image

# Import from utils.py
from utils import read_file_bytes


# Store receipts in memory
RECEIPTS: Dict[str, Receipt] = {}

# ---------------- Config ----------------
CHAT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SYSTEM_PROMPT = (
    "You are a helpful assistant for Swedish shopping receipts. "
    "Only answer using the receipts context below. Prices are SEK unless stated. "
    "If unsure, say you don't know."
)

MAX_CTX_JSON_CHARS = 6000

# Init chat model (GPU if available)
tok = AutoTokenizer.from_pretrained(CHAT_MODEL)
mdl = AutoModelForCausalLM.from_pretrained(CHAT_MODEL)
chat = pipeline("text-generation", model=mdl, tokenizer=tok, max_new_tokens=256, temperature=0.2, top_p=0.9)


# Build context
def build_context() -> str:
    """Build context from both in-memory receipts and database."""
    rows = []

    # Add in-memory receipts
    for r in RECEIPTS.values():
        rows.append({
            "receipt_id": r.receipt_id,
            "store_name": r.store_name,
            "date": r.date,
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
            "store_name": r["store_name"],
            "date": r["date"],
            "items": r["items"]
        })

    s = json.dumps(rows, ensure_ascii=False)
    return s[:MAX_CTX_JSON_CHARS]


# Chat respond 
def chat_respond(message, history):
    """Respond to chat messages using LLM."""
    # Initialize history as empty list if None
    if history is None:
        history = []
    
    if not RECEIPTS and not get_all_receipts_from_db():
        reply = "I have no receipts yet. Please upload a PDF or image first."
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": reply})
        return history, ""
    
    ctx = build_context()
    transcript = ""
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        transcript += f"{role}: {msg['content']}\n"
    
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"RECEIPTS_CONTEXT:\n{ctx}\n\n"
        f"{transcript}"
        f"User: {message}\nAssistant:"
    )
    
    out = chat(prompt)[0]["generated_text"]
    reply = out.split("Assistant:")[-1].strip()
    
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    
    return history, ""
    
# Clear all
def clear_all():
    """Wrapper to clear database and in-memory receipts."""
    clear_db()
    RECEIPTS.clear()
    return "Database and receipts cleared successfully!"


# Do extract
def do_extract(files):
    """Extract receipts from uploaded files."""
    results = []
    for f in files:
        try:
            data, name = read_file_bytes(f)
            
            # Parse based on file type
            if name.lower().endswith(".pdf"):
                rec = extract_from_pdf(data, name)
            else:
                rec = extract_from_image(data, name)

            # Save to database
            stats = save_receipt_to_db(rec)

            # Keep in memory too
            RECEIPTS[rec.receipt_id] = rec

            result = rec.model_dump()
            result["db_stats"] = stats
            results.append(result)
        except Exception as e:
            print("[extract error]", name if 'name' in locals() else f, "->", e)
            traceback.print_exc()
            results.append({"file": name if 'name' in locals() else str(f), "error": str(e)})
    
    return results

# Save receipt to database
def save_receipt_to_db(receipt: Receipt):
    
    # Add store (or get existing store_id)
    store_id, store_existed = add_store_to_db(
        name=receipt.store_name,
        address=receipt.address,
        post_address=receipt.post_address,
        short_name=None,  
        phone_number=None,  
        org_number=None,  
    )
    receipt.store_id = store_id

    # Add receipt
    receipt_id, receipt_existed = add_receipt_to_db(
        store_id=store_id,
        date=receipt.date,
        total=receipt.total,
        store_existed=store_existed,
    )
    receipt.receipt_id = receipt_id 

    if not receipt_existed:
        purchase_date = receipt.date or datetime.now().date().isoformat() # Also stored in receipt table
        
        for item in receipt.line_items:
            # Calculate final price and discount
            final_total = item.line_total_after_discount if item.is_discounted else item.line_total
            discount_amount = abs(item.discount_amount_total) if item.discount_amount_total else 0.0
            
            _, was_created = add_item_to_db(
                description=item.description,
                article_number=item.item_code,
                price=item.unit_price,
                quantity=item.qty,
                total=final_total,
                discount=discount_amount,
                category=None,  # We don't extract category yet
                store_id=store_id,
                receipt_id=receipt_id,
                purchase_date=purchase_date, # Also stored in receipt table
                comparison_price=None,
                comparison_price_unit=None
            )
    
    stats = {"new": 0, "existing": 0}
    return stats    

# Gradio Interface
with gr.Blocks(title="Receipt Analyzer") as gradio_ux:
    gr.Markdown("## 🧾 Receipt Analyzer\nUpload PDFs or images → extract → auto-save to database → chat about prices")

    with gr.Tab("1) Upload & Extract"):
        files = gr.File(
            file_count="multiple",
            file_types=[".pdf", ".png", ".jpg", ".jpeg"],
            label="Upload receipts"
        )
        out = gr.JSON(label="Parsed receipts (JSON) - includes DB stats")
        gr.Button("Extract & Save to DB").click(do_extract, inputs=files, outputs=out)

    with gr.Tab("2) Chat"):
        chatbot = gr.Chatbot(label="Chat", height=420, type="messages")
        msg = gr.Textbox(
            label="Ask (press Enter to send)",
            placeholder="e.g., What did milk cost at ICA in June?",
            lines=1
        )
        with gr.Row():
            send = gr.Button("Send", variant="primary")
            clear = gr.Button("Clear")
        
        send.click(chat_respond, inputs=[msg, chatbot], outputs=[chatbot, msg])
        msg.submit(chat_respond, inputs=[msg, chatbot], outputs=[chatbot, msg])
        clear.click(lambda: ([], ""), outputs=[chatbot, msg], queue=False)

    with gr.Tab("3) Database View"):
        gr.Markdown("### View stored receipts and items")
        db_output = gr.Markdown()
        with gr.Row():
            refresh_btn = gr.Button("Refresh View", variant="primary")
            clear_db_btn = gr.Button("Clear Database", variant="stop")

        refresh_btn.click(view_db, outputs=db_output)
        clear_db_btn.click(clear_all, outputs=db_output)

        # Load database view on tab open
        gradio_ux.load(view_db, outputs=db_output)


# Init/Main
if __name__ == "__main__":
    # Initialize database on startup
    print("Init database...")
    init_db()
    print("[run] starting Gradio on http://127.0.0.1:7860 …")
    gradio_ux.launch(server_name="127.0.0.1", server_port=7860, share=False)
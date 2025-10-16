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
    get_stats_from_db, get_recent_items, clear_db, search_items, view_db, get_number_of_receipts_in_db
)

# Import from parser.py
from parser import Receipt, extract_from_pdf, extract_from_image

# Import from utils.py
from utils import read_file_bytes

# ---------------- Config ----------------
CHAT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

SYSTEM_PROMPT = (
    "You are a helpful assistant for Swedish shopping receipts. "
    "Only answer using the receipts context below. Prices are SEK unless stated. "
    "If unsure, say you don't know."
)

# Human-readable schema/instruction for the receipts JSON that is prepended to the LLM prompt.
RECEIPTS_SCHEMA = (
    "Receipts JSON structure:\n"
    "- A list of receipts. Each receipt object has:\n"
    "  - receipt_id: int\n"
    "  - store_id: int | null\n"
    "  - store_name: string\n"
    "  - date: string (ISO date, YYYY-MM-DD)\n"
    "  - total: number (receipt total, in SEK)\n"
    "  - items: list of item objects\n"
    "    - item object fields:\n"
    "      - item_id: int | null\n"
    "      - description: string\n"
    "      - article_number: string | null\n"
    "      - price: number (unit price, SEK)\n"
    "      - quantity: number\n"
    "      - total: number (line total, SEK)\n"
    "      - discount: number | null (discount amount on line)\n"
    "Only use these fields from the RECEIPTS_CONTEXT when answering. Do not invent values."
)

MAX_CTX_JSON_CHARS = 2048

# Init chat model (GPU if available)
tok = AutoTokenizer.from_pretrained(CHAT_MODEL)
mdl = AutoModelForCausalLM.from_pretrained(CHAT_MODEL)
chat = pipeline("text-generation", model=mdl, tokenizer=tok, max_new_tokens=256, temperature=0.2, top_p=0.9)


# Build context
def build_context() -> str:
    """Build context from both in-memory receipts and database."""
    rows = []

    # Add database receipts
    db_receipts = get_all_receipts_from_db()
    
    s = json.dumps(db_receipts, ensure_ascii=False, default=str, indent=2)
    print(s[:MAX_CTX_JSON_CHARS])
    return s[:MAX_CTX_JSON_CHARS]

# Chat respond 
def chat_respond(message, history):
    """Respond to chat messages using LLM.

    - Accepts history as either list-of-tuples [(user, assistant), ...] (Gradio typical)
      or list-of-dicts [{"role":"user"/"assistant","content": "..."}, ...].
    - Preserves the incoming history format when returning the updated history.
    - Prepends a small human-readable schema before the receipts JSON so the LLM
      understands the structure.
    """
    # Normalize history container
    if history is None:
        history = []

    # If no receipts, return helpful message without calling the model
    if get_number_of_receipts_in_db() == 0:
        reply = "I have no receipts yet. Please upload a PDF or image first."
        # preserve incoming format when returning
        if len(history) > 0 and isinstance(history[0], (list, tuple)):
            new_history = history.copy()
            new_history.append((message, reply))
            return new_history, ""
        else:
            new_history = history.copy()
            new_history.append({"role": "user", "content": message})
            new_history.append({"role": "assistant", "content": reply})
            return new_history, ""

    # Build transcript from history (works for both formats)
    transcript = ""
    incoming_is_tuple_pairs = False
    if len(history) > 0 and isinstance(history[0], (list, tuple)):
        incoming_is_tuple_pairs = True
        for u, a in history:
            if u:
                transcript += f"User: {u}\n"
            if a:
                transcript += f"Assistant: {a}\n"
    else:
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                transcript += f"User: {content}\n"
            else:
                transcript += f"Assistant: {content}\n"

    # Build prompt with schema + receipts context
    ctx = build_context()
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"{RECEIPTS_SCHEMA}\n\n"
        f"RECEIPTS_CONTEXT:\n{ctx}\n\n"
        f"{transcript}"
        f"User: {message}\nAssistant:"
    )

    # Call LLM (protect with try/except)
    try:
        out = chat(prompt)[0]["generated_text"]
        reply = out.split("Assistant:")[-1].strip()
    except Exception as e:
        # On error, return a short error reply and preserve history format
        err_msg = f"[LLM error] {str(e)}"
        if incoming_is_tuple_pairs:
            new_history = history.copy()
            new_history.append((message, err_msg))
            return new_history, ""
        else:
            new_history = history.copy()
            new_history.append({"role": "user", "content": message})
            new_history.append({"role": "assistant", "content": err_msg})
            return new_history, ""

    # Append to history in same format it was received
    if incoming_is_tuple_pairs:
        new_history = history.copy()
        new_history.append((message, reply))
    else:
        new_history = history.copy()
        new_history.append({"role": "user", "content": message})
        new_history.append({"role": "assistant", "content": reply})

    # Return updated chatbot history and clear the input textbox
    return new_history, ""
    
# Clear all
def clear_all():
    """Wrapper to clear database"""
    clear_db()
    return "Database cleared!"

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
            receipt_existed = save_receipt_to_db(rec)

            # Keep in memory too
            # RECEIPTS[rec.receipt_id] = rec

            result = rec.model_dump()
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

    # Add items
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
    
    return receipt_existed    

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
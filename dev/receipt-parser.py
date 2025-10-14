import os
import json
import gradio as gr
import ollama
from pdf2image import convert_from_path
from PIL import Image
import pytesseract

# ===============================
# KONFIG
# ===============================
# Om Tesseract inte finns i PATH, ange full väg:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Använd större modell för bättre förståelse
TEXT_MODEL = "qwen2.5:7b"  # <-- Ändrad från 3b till 7b

# ===============================
# FÖRBÄTTRAD SYSTEM PROMPT
# ===============================
SYSTEM_PROMPT = """You are an expert at analyzing Swedish receipt text and converting it to structured JSON.

INSTRUCTIONS FOR READING THE RECEIPT:

1. STORE NAME: Look at the top - find the actual business name (e.g., "ICA Bonden", "Willys", "Coop")
   - Ignore words like "Kvitto", "Receipt", "Datum", "Tid"

2. DATE & TIME: Look for patterns like:
   - "Datum 2025-10-04" or "2025-10-04"
   - "Tid 16:38" or time stamps

3. ITEMS: This is the most important part!
   - Each item has: Description, Article number, Unit price, Quantity, Total price
   - Look for lines with product names followed by numbers
   - The LAST number on each item line is usually the TOTAL PRICE (what was paid)
   - Format example: "Cheez doodles Olw    1001443    13,95    1,00 st    13,95"
     This means: Name="Cheez doodles Olw", quantity=1, price=13.95 (the last number)
   
4. DISCOUNTS/RABATT: Lines with negative numbers (like "-7,90") are discounts
   - These are already included in the total, don't add them as separate items
   - Lines like "Chips/Doodle 2f20kr" describe the discount offer

5. TOTALS: Look for:
   - "Betalat" or "Totalt" = Total amount paid
   - "Moms" = Tax/VAT
   - The FINAL total is what the customer paid

6. NUMBERS: Swedish format uses comma for decimals
   - "13,95" = 13.95
   - "134,85" = 134.85

CRITICAL: Extract ALL items from the receipt. Count how many product lines you see and make sure you include them all.

OUTPUT FORMAT (JSON only):
{
    "store_name": "Store Name",
    "date": "YYYY-MM-DD",
    "time": "HH:MM",
    "items": [
        {"name": "Product name", "quantity": 1, "price": 13.95}
    ],
    "subtotal": 100.00,
    "tax": 12.00,
    "total": 112.00,
    "payment_method": "card",
    "receipt_number": "584"
}

IMPORTANT:
- Include EVERY product line you see
- Use the LAST price number for each item (that's what was actually paid)
- Don't include discount description lines as items
- Total should be the large number near "Betalat" or "Totalt"
- Use decimals with dots (13.95 not 13,95)

Only respond with valid JSON, nothing else."""

# ===============================
# OCR FUNKTION
# ===============================
def extract_text_from_image(image: Image.Image) -> str:
    """Extract text from image using Tesseract OCR"""
    try:
        # Tesseract with Swedish language support
        # If Swedish doesn't work, remove lang='swe' or use lang='eng'
        text = pytesseract.image_to_string(image, lang='swe+eng')
        return text
    except Exception as e:
        print(f"OCR Error: {e}")
        # Fallback to English only
        text = pytesseract.image_to_string(image, lang='eng')
        return text

# ===============================
# PROCESS FUNKTION
# ===============================
def process_pdf_receipt(pdf_file):
    """Process receipt: PDF -> Image -> OCR -> LLM -> JSON"""
    
    if pdf_file is None:
        return None, "❌ Please upload a PDF file first!"
    
    try:
        print(f"Processing file: {pdf_file.name}")
        
        # Step 1: Convert PDF to image
        print("Converting PDF to image...")
        images = convert_from_path(pdf_file.name, first_page=1, last_page=1, dpi=300)
        receipt_image = images[0]
        print(f"✓ PDF converted (size: {receipt_image.size})")
        
        # Step 2: Extract text with Tesseract OCR
        print("Extracting text with Tesseract OCR...")
        ocr_text = extract_text_from_image(receipt_image)
        
        print("\n" + "="*50)
        print("OCR TEXT EXTRACTED:")
        print("="*50)
        print(ocr_text)
        print("="*50 + "\n")

        # Räkna items för debugging
        item_lines = [line for line in ocr_text.split('\n') if any(word in line.lower() for word in ['kr', 'sek', ',']) and len(line) > 10]
        print(f"DEBUG: Found approximately {len(item_lines)} potential item lines")
        
        if not ocr_text.strip():
            return None, "❌ No text found in image. Try a clearer scan."
        
        # Step 3: Send text to LLM for structuring
        print(f"Sending to {TEXT_MODEL} for JSON structuring...")
        
        prompt = f"{SYSTEM_PROMPT}\n\nHere is the OCR text from the receipt:\n\n{ocr_text}\n\nNow extract the data as JSON:"
        
        response = ollama.chat(
            model=TEXT_MODEL,
            messages=[{
                'role': 'user',
                'content': prompt
            }],
            options={
                'temperature': 0.1,
                'num_predict': 1000
            }
        )
        
        print(f"✓ Response received from {TEXT_MODEL}")
        
        # Step 4: Parse JSON
        json_text = response['message']['content'].strip()
        
        print(f"\nRaw LLM response:\n{json_text}\n")
        
        # Remove markdown formatting
        if json_text.startswith("```json"):
            json_text = json_text[7:]
        if json_text.startswith("```"):
            json_text = json_text[3:]
        if json_text.endswith("```"):
            json_text = json_text[:-3]
        
        json_text = json_text.strip()
        
        # Parse JSON
        receipt_data = json.loads(json_text)
        
        # Print result
        print("\n" + "="*50)
        print("EXTRACTED RECEIPT DATA:")
        print("="*50)
        print(json.dumps(receipt_data, indent=2, ensure_ascii=False))
        print("="*50 + "\n")
        
        return receipt_data, f"✅ Receipt processed successfully!\n(OCR: Tesseract, LLM: {TEXT_MODEL})"
        
    except json.JSONDecodeError as e:
        error_msg = f"❌ Failed to parse JSON: {str(e)}\n\nRaw response:\n{json_text}"
        print(error_msg)
        return None, error_msg
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return None, error_msg

# ===============================
# GRADIO INTERFACE
# ===============================
with gr.Blocks(title="Receipt Parser - Tesseract + Ollama") as demo:
    gr.Markdown("# 🧾 Receipt JSON Extractor")
    gr.Markdown(f"**OCR:** Tesseract | **LLM:** {TEXT_MODEL}")
    gr.Markdown("**100% Free - No API costs!** 🆓")
    
    with gr.Row():
        with gr.Column():
            pdf_input = gr.File(
                label="Upload Receipt PDF", 
                file_types=[".pdf"],
                type="filepath"
            )
            process_btn = gr.Button("📤 Extract Data", variant="primary")
            
            gr.Markdown("""
            **How it works:**
            1. PDF → Image conversion
            2. Tesseract OCR extracts text
            3. Local LLM structures it as JSON
            """)
        
        with gr.Column():
            status_output = gr.Textbox(label="Status", lines=3)
            json_output = gr.JSON(label="Extracted Data")
    
    process_btn.click(
        fn=process_pdf_receipt,
        inputs=[pdf_input],
        outputs=[json_output, status_output]
    )
    
    gr.Markdown("---")
    gr.Markdown("**Powered by:** Tesseract OCR (free) + Ollama (free) | **GPU:** RTX 3060")

if __name__ == "__main__":
    print("Starting Receipt Parser (Tesseract OCR + Ollama)...")
    print(f"Using model: {TEXT_MODEL}")
    print("This solution is 100% free!")
    demo.launch(share=False)
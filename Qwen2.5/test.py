"""Utility script for extracting structured product data from a PDF using Qwen2.5-VL."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

try:
    import pypdfium2 as pdfium
except ImportError as exc:  # pragma: no cover - user must install dependency
    raise SystemExit(
        "pypdfium2 is required for rendering PDF pages to images. Install it via 'pip install pypdfium2'."
    ) from exc

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract product price data from a PDF with Qwen2.5-VL")
    parser.add_argument("pdf", type=Path, help="Path to the PDF file to analyse")
    parser.add_argument("output", type=Path, help="Path to the JSON file that will store the structured data")
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=(
            "You are a meticulous retail analyst. For the provided PDF pages, extract a JSON array named "
            "'products'. Each element must have the keys 'name', 'price', and optionally 'discount'. "
            "Ignore decorative text, headers, footers, and descriptions without prices."
        ),
        help="System prompt that sets the behaviour of the assistant.",
    )
    parser.add_argument(
        "--user-prompt",
        type=str,
        default=(
            "Analyse the catalogue pages and list every product with its price and any discount information. "
            "Return only valid JSON with a top-level 'products' array."
        ),
        help="Instruction appended to the user message in the conversation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate for the response.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature used during generation.",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device map passed to the model loader. Use 'auto' to leverage available accelerators.",
    )
    return parser.parse_args()


def render_pdf_to_images(pdf_path: Path) -> List[Any]:
    """Render all pages of the PDF to PIL images."""
    LOGGER.info("Rendering PDF: %s", pdf_path)
    doc = pdfium.PdfDocument(str(pdf_path))
    images: List[Any] = []
    for page_index in range(len(doc)):
        page = doc.get_page(page_index)
        try:
            bitmap = page.render(scale=2, rotation=0)
            image = bitmap.to_pil()
            images.append(image)
        finally:
            page.close()
    if not images:
        raise ValueError(f"No pages were rendered from PDF: {pdf_path}")
    LOGGER.info("Rendered %d page(s)", len(images))
    return images


def load_model(model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ", device_map: str = "auto"):
    LOGGER.info("Loading processor and model: %s", model_id)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
        trust_remote_code=True,
    )
    return processor, model


def build_conversation(system_prompt: str, user_prompt: str, images: List[Any]) -> List[Dict[str, Any]]:
    """Build a multi-turn conversation for Qwen2.5-VL that includes all PDF images."""
    conversation: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        }
    ]

    user_content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for index, image in enumerate(images, start=1):
        user_content.append(
            {"type": "text", "text": f"<page {index}>"}
        )
        user_content.append({"type": "image", "image": image})

    conversation.append({"role": "user", "content": user_content})
    return conversation


def generate_response(
    processor: AutoProcessor,
    model: Qwen2VLForConditionalGeneration,
    conversation: List[Dict[str, Any]],
    max_new_tokens: int,
    temperature: float,
) -> str:
    LOGGER.info("Generating structured response from the model")
    inputs = processor(
        conversation,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
    )
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    LOGGER.debug("Raw model response: %s", response)
    return response.strip()


def extract_json(response_text: str) -> Dict[str, Any]:
    """Attempt to parse the model response into JSON."""
    LOGGER.info("Parsing model response into JSON")

    # Try direct parsing first.
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        LOGGER.debug("Direct JSON parsing failed, attempting to locate JSON substring")

    # Fallback: locate the first JSON object/array inside the response.
    start = response_text.find("{")
    if start == -1:
        start = response_text.find("[")
    if start == -1:
        raise ValueError("Model response does not contain JSON data")

    # Attempt to parse progressively larger substrings to capture complete JSON.
    for end in range(len(response_text), start, -1):
        snippet = response_text[start:end]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            continue
    raise ValueError("Unable to parse JSON from the model response")


def save_json(data: Dict[str, Any], output_path: Path) -> None:
    LOGGER.info("Saving JSON output to %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    if not args.pdf.exists():
        raise FileNotFoundError(f"PDF not found: {args.pdf}")

    images = render_pdf_to_images(args.pdf)
    processor, model = load_model(device_map=args.device_map)
    conversation = build_conversation(args.system_prompt, args.user_prompt, images)
    response_text = generate_response(
        processor=processor,
        model=model,
        conversation=conversation,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    json_output = extract_json(response_text)
    save_json(json_output, args.output)


if __name__ == "__main__":
    main()

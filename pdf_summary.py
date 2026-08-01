import argparse
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise SystemExit("OPENROUTER_API_KEY is missing. Add it to .env and try again.")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

MODEL = "qwen/qwen3.5-flash-02-23"

# Model has a ~32K context window. Reserve ~2K for system prompt + output.
# One token ≈ 4 characters for English, so 30K tokens ≈ 120K characters.
MAX_INPUT_CHARS = 120_000

SYSTEM_PROMPT = """You are a precise document-summarization assistant. You will receive the full text of a PDF document, with pages labeled [Page 1], [Page 2], etc.

Output exactly three sections with these headings:

Overview
- A 2-3 sentence summary of what the document is about.

Key Points
- Bullet points of the most important facts or ideas from the document.
- Every bullet point must end with a [Page X] citation showing which page the information came from.

Limitations
- 1-2 sentences noting what the document does NOT cover or what questions it leaves unanswered.

Rules:
- Use ONLY information from the provided text.
- Do not use outside knowledge or make assumptions beyond the text.
- Every Key Point must include a [Page X] citation."""


def extract_pages(path: str) -> list[dict]:
    """Extract text from each page of a PDF. Returns [{page_num, text}]."""
    try:
        reader = PdfReader(path)
    except FileNotFoundError:
        print(f"Error: File not found: '{path}'")
        sys.exit(1)
    except IsADirectoryError:
        print(f"Error: '{path}' is a directory, not a file.")
        sys.exit(1)
    except PermissionError:
        print(f"Error: Permission denied to read '{path}'.")
        sys.exit(1)
    except Exception as exc:
        print(f"Error: Could not read PDF: {exc}")
        sys.exit(1)

    pages = []
    total = len(reader.pages)
    for i, page in enumerate(reader.pages, start=1):
        print(f"Extracting page {i}/{total}...")
        text = page.extract_text() or ""
        pages.append({"page_num": i, "text": text.strip()})
    return pages


def has_extractable_text(pages: list[dict]) -> bool:
    """Return True if at least one page has meaningful text content."""
    total = sum(len(p["text"]) for p in pages)
    return total > 50


def build_prompt(pages: list[dict]) -> str:
    """Build the user prompt from extracted pages."""
    labeled = []
    for p in pages:
        labeled.append(f"[Page {p['page_num']}]\n{p['text']}")
    return "\n\n".join(labeled)


def summarize(text: str) -> str:
    """Send the document text to the LLM and return the summary."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        max_tokens=600,
    )
    return response.choices[0].message.content


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize a PDF document with page-level citations."
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to the PDF file to summarize.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pages = extract_pages(args.path)

    if not has_extractable_text(pages):
        print(
            "This PDF contains no extractable text. "
            "It may be a scanned document (images only). "
            "OCR is not supported in this version."
        )
        sys.exit(0)

    prompt = build_prompt(pages)

    if len(prompt) > MAX_INPUT_CHARS:
        approx_tokens = len(prompt) // 4
        print(
            f"Error: The extracted text is too long ({len(prompt):,} chars, "
            f"roughly {approx_tokens:,} tokens).\n"
            f"The model supports up to ~{MAX_INPUT_CHARS:,} chars of input.\n\n"
            f"This PDF has {len(pages)} pages. Try one of these:\n"
            f"  - Split the PDF into smaller files (e.g., 10 pages each)\n"
            f"  - Use a model with a larger context window\n"
            f"  - Summarize each chapter separately"
        )
        sys.exit(1)

    summary = summarize(prompt)
    print(summary)


if __name__ == "__main__":
    main()

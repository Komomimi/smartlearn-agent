import argparse
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise SystemExit("OPENROUTER_API_KEY is missing. Add it to .env and try again.")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

MODEL = "qwen/qwen3.5-flash-02-23"

SYSTEM_PROMPT = """You are a precise reading-comprehension assistant. You will receive:
1. A text split into numbered paragraphs like [Paragraph 1], [Paragraph 2], etc.
2. A question about that text.

Rules:
- Answer the question using ONLY information from the provided text.
- For every factual claim in your answer, cite the paragraph it came from using [Paragraph X] notation.
- If the text does not contain the answer, respond EXACTLY: The text does not provide this information.
- Do not use outside knowledge or make assumptions beyond the text."""


def read_input_text() -> str:
    """Read multi-line text from stdin until a line containing only 'END'."""
    print("Paste your text below. Type 'END' on a new line when finished:")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines)


def read_file_text(path: str) -> str:
    """Read text from a file, with a friendly error if it fails."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found: '{path}'")
        sys.exit(1)
    except IsADirectoryError:
        print(f"Error: '{path}' is a directory, not a file.")
        sys.exit(1)
    except PermissionError:
        print(f"Error: Permission denied to read '{path}'.")
        sys.exit(1)


def split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs by blank lines, keeping only non-empty ones."""
    raw = text.split("\n\n")
    return [p.strip() for p in raw if p.strip()]


def build_labeled_text(paragraphs: list[str]) -> str:
    """Prepend [Paragraph N] labels to each paragraph."""
    labeled = []
    for i, para in enumerate(paragraphs, start=1):
        labeled.append(f"[Paragraph {i}] {para}")
    return "\n\n".join(labeled)


def ask(text: str, question: str) -> str:
    """Send the labeled text and question to the LLM, return the answer."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Text:\n{text}\n\nQuestion: {question}"},
        ],
        max_tokens=500,
    )
    return response.choices[0].message.content


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask questions about a text with paragraph-level citations."
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a text file to load instead of pasting interactively.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.file:
        raw_text = read_file_text(args.file)
    else:
        raw_text = read_input_text()

    if not raw_text.strip():
        print("Error: No text provided. Please paste some text and try again.")
        sys.exit(1)

    paragraphs = split_paragraphs(raw_text)
    if not paragraphs:
        print("Error: No text provided. Please paste some text and try again.")
        sys.exit(1)

    labeled_text = build_labeled_text(paragraphs)
    source = f"file '{args.file}'" if args.file else "pasted text"
    print(f"\nText loaded ({len(paragraphs)} paragraph(s)) from {source}. Ask questions below.")
    print("Type 'quit' or press Ctrl+C to exit.\n")

    while True:
        try:
            question = input("Your question: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question.strip():
            continue

        if question.strip().lower() in ("quit", "exit", "q"):
            break

        answer = ask(labeled_text, question)
        print(f"\n{answer}\n")


if __name__ == "__main__":
    main()

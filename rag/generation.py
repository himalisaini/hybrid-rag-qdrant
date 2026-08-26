"""Generate a cited answer from retrieved chunks using the Gemini API.

Rotates across several free-tier API keys (GEMINI_API_KEYS, comma-separated)
so that quota is spread across keys rather than exhausting one.
"""

import itertools
import os

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors

from rag.retrieval import SearchResult

load_dotenv()

MODEL = "gemini-3.7-flash"

SYSTEM_PROMPT = (
    "You are a documentation assistant. Answer the user's question using ONLY "
    "the passages provided below -- do not use outside knowledge. Every claim "
    "must be followed by a citation to the passage it came from, in square "
    "brackets, e.g. [tutorial/query-params.md]. If the passages don't contain "
    "enough information to answer, say so plainly instead of guessing."
)

_keys = [k.strip() for k in os.environ["GEMINI_API_KEYS"].split(",") if k.strip()]
_key_cycle = itertools.cycle(_keys)


def _format_context(results: list[SearchResult]) -> str:
    blocks = [f"[{r.chunk.chunk_id}]\n{r.chunk.text}" for r in results]
    return "\n\n---\n\n".join(blocks)


def generate_answer(query: str, results: list[SearchResult]) -> str:
    context = _format_context(results)
    user_message = f"Passages:\n\n{context}\n\nQuestion: {query}"

    last_error: Exception | None = None
    for _ in range(len(_keys)):
        key = next(_key_cycle)
        client = genai.Client(api_key=key)
        try:
            interaction = client.interactions.create(
                model=MODEL,
                system_instruction=SYSTEM_PROMPT,
                input=user_message,
            )
            return interaction.output_text
        except genai_errors.ClientError as e:
            last_error = e
            continue

    raise RuntimeError(f"All {len(_keys)} Gemini API keys failed. Last error: {last_error}")

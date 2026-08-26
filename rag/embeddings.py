"""Thin wrapper around the Voyage AI embedding API.

Rate-limited for Voyage's no-payment-method free tier: 3 requests/minute,
10K tokens/minute. Batches are capped well under that per-request budget,
and calls are spaced >=21s apart so 3 calls/minute is never exceeded.
"""

import os

import tiktoken
import voyageai
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm

from rag.rate_limit import wait_for_rate_limit

load_dotenv()

MODEL = "voyage-3.5"
EMBED_DIM = 1024  # fixed output size of voyage-3.5

MAX_BATCH_TOKENS = 2800
MAX_BATCH_SIZE = 40

_ENC = tiktoken.get_encoding("cl100k_base")  # proxy tokenizer, just for batch sizing
_client = None


def _get_client() -> voyageai.Client:
    global _client
    if _client is None:
        _client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    return _client


def _make_batches(texts: list[str]) -> list[list[str]]:
    batches: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for text in texts:
        t = len(_ENC.encode(text))
        if current and (current_tokens + t > MAX_BATCH_TOKENS or len(current) >= MAX_BATCH_SIZE):
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(text)
        current_tokens += t
    if current:
        batches.append(current)
    return batches


@retry(
    retry=retry_if_exception_type(voyageai.error.APIConnectionError),
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=2, min=2, max=60),
)
def _embed_batch(client: voyageai.Client, batch: list[str], input_type: str):
    return client.embed(batch, model=MODEL, input_type=input_type)


def embed_texts(texts: list[str], input_type: str) -> list[list[float]]:
    """input_type is 'document' when indexing chunks, 'query' when embedding a search question."""
    client = _get_client()
    batches = _make_batches(texts)
    vectors: list[list[float]] = []
    for batch in tqdm(batches, desc=f"embedding ({input_type}, {len(batches)} batches)"):
        wait_for_rate_limit()
        result = _embed_batch(client, batch, input_type)
        vectors.extend(result.embeddings)
    return vectors

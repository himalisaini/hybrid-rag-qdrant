"""Shared rate limiter for Voyage AI calls. Embeddings and reranking are separate
API endpoints but draw on the same account-wide free-tier quota, so they must
share one clock -- two independent 21s-apart limiters could each fire within
the other's window and blow the real combined limit.
"""

import time

MIN_SECONDS_BETWEEN_CALLS = 21
_last_call_time = 0.0


def wait_for_rate_limit() -> None:
    global _last_call_time
    elapsed = time.monotonic() - _last_call_time
    if elapsed < MIN_SECONDS_BETWEEN_CALLS:
        time.sleep(MIN_SECONDS_BETWEEN_CALLS - elapsed)
    _last_call_time = time.monotonic()

import os
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

MAX_REQUESTS = 60
WINDOW_SECONDS = 60
BLOCKED_IPS = set(filter(None, os.getenv("BLOCKED_IPS", "").split(",")))

_hits = defaultdict(deque)


def rate_limit(request: Request):
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        ip = forwarded.split(",")[0].strip()
    else:
        ip = request.client.host if request.client else "unknown"

    if ip in BLOCKED_IPS:
        raise HTTPException(status_code=403, detail="IP blocked")

    now = time.time()
    hits = _hits[ip]
    while hits and now - hits[0] > WINDOW_SECONDS:
        hits.popleft()
    if len(hits) >= MAX_REQUESTS:
        raise HTTPException(status_code=429, detail="Too many requests. Try again later.")
    hits.append(now)
"""Runs the NLP server."""

import asyncio
import logging
from typing import Optional

from fastapi import FastAPI, Request
from nlp_manager import NLPManager

app = FastAPI()
manager = NLPManager()
logger = logging.getLogger(__name__)


class _LoadState:
    """Tracks corpus-loading state for async, pollable behavior."""

    def __init__(self) -> None:
        self.status: str = "idle"  # idle | loading | loaded | failed
        self.error: Optional[str] = None
        self.task: Optional[asyncio.Task] = None
        self.lock = asyncio.Lock()


load_state = _LoadState()


def _do_load(documents) -> bool:
    """Synchronous corpus load. Runs on a worker thread."""
    manager.load_corpus(documents)
    return manager.loaded


async def _load_task(documents) -> None:
    try:
        ok = await asyncio.to_thread(_do_load, documents)
        load_state.status = "loaded" if ok else "failed"
    except Exception as e:
        logger.exception("Corpus load failed")
        load_state.status = "failed"
        load_state.error = str(e)


@app.post("/nlp")
async def nlp(request: Request) -> dict[str, list[dict[str, list[str] | str]]]:
    inputs_json = await request.json()
    first = inputs_json["instances"][0]

    # Load: any request carrying `documents` kicks off the load.
    if first.get("documents") is not None:
        async with load_state.lock:
            if load_state.status == "idle":
                load_state.status = "loading"
                load_state.task = asyncio.create_task(_load_task(first["documents"]))
            return {"predictions": [{"status": load_state.status}]}
    # Poll: returns current status (subsequent polls).
    if first.get("poll") is not None:
        return {"predictions": [{"status": load_state.status}]}

    predictions = [
        await asyncio.to_thread(manager.qa, instance["question"])
        for instance in inputs_json["instances"]
    ]

    return {"predictions": predictions}


@app.get("/health")
def health() -> dict[str, str]:
    return {"message": "health ok"}

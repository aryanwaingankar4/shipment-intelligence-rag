"""ChromaDB access.

- CHROMA_HOST set   -> HttpClient to a Chroma server (shared by many API replicas).
- CHROMA_HOST unset -> embedded PersistentClient on local disk (dev / single box).
"""

import logging

from tenacity import retry, stop_after_attempt, wait_exponential

from shipment_rag.config import Settings

log = logging.getLogger(__name__)


@retry(stop=stop_after_attempt(10), wait=wait_exponential(min=1, max=10), reraise=True)
def get_client(settings: Settings):
    import chromadb

    if settings.chroma_host:
        client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
        client.heartbeat()  # fail fast (and retry) if the server isn't up yet
        log.info("connected to chroma server", extra={"host": settings.chroma_host})
        return client
    settings.chroma_persist_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(settings.chroma_persist_path))


def get_collection(settings: Settings, client=None):
    client = client or get_client(settings)
    return client.get_or_create_collection(
        name=settings.collection_name, metadata={"hnsw:space": "cosine"}
    )

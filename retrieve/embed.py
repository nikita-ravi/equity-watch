"""Query embedding -- the same model, loaded once per process.

Deliberately thin: it reuses ingest.embed.Embedder rather than defining a second
model wrapper, so query vectors can never drift from the vectors in Qdrant.
"""

import config
from ingest.embed import Embedder

_embedder = None


def get_embedder():
    """Return the process-wide Embedder, loading the model on first use only."""
    global _embedder
    if _embedder is None:
        _embedder = Embedder(config.EMBEDDING_MODEL, config.EMBEDDING_BATCH_SIZE)
    return _embedder


def embed_query(query):
    """Embed a single query string, applying the model's query instruction.

    bge is asymmetric: passages were embedded bare at ingest, queries take the
    instruction prefix. config.QUERY_INSTRUCTION = "" turns this off.
    """
    embedder = get_embedder()
    text = f"{config.QUERY_INSTRUCTION}{query}" if config.QUERY_INSTRUCTION else query
    return embedder.embed([text])[0]

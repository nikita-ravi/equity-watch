"""Qdrant collection management and idempotent upserts."""

import logging
import uuid

import config

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

log = logging.getLogger(__name__)

# Stable namespace so the same chunk always maps to the same point ID across runs.
_NAMESPACE = uuid.UUID("6f0b6a1e-5d1d-4b8e-9f0a-1c2d3e4f5a6b")


def get_client(url):
    return QdrantClient(url=url, timeout=config.QDRANT_TIMEOUT)


def ensure_collection(client, name, dim):
    """Create the collection if absent; leave an existing one untouched."""
    if client.collection_exists(name):
        info = client.get_collection(name)
        existing = info.config.params.vectors.size
        if existing != dim:
            raise RuntimeError(
                f"collection '{name}' already exists with vector size {existing}, "
                f"but the embedding model produces {dim}. Drop the collection or "
                f"use a different collection name."
            )
        log.info("collection '%s' already exists (size=%d) -- skipping creation",
                 name, existing)
        return False

    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    log.info("created collection '%s' (size=%d, distance=Cosine)", name, dim)
    return True


def point_id(filing_id, section, chunk_index):
    """Deterministic point ID -> re-ingesting a filing overwrites, never duplicates.

    chunk_index is per-section, so the section must be part of the key; filing_id
    plus chunk_index alone would collide across sections.
    """
    return str(uuid.uuid5(_NAMESPACE, f"{filing_id}|{section}|{chunk_index}"))


def upsert_chunks(client, name, records, batch_size=256):
    """Upsert (payload, vector) records. Returns the number of points written."""
    total = 0
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        points = [
            PointStruct(
                id=point_id(r["payload"]["filing_id"],
                            r["payload"]["section"],
                            r["payload"]["chunk_index"]),
                vector=r["vector"],
                payload=r["payload"],
            )
            for r in batch
        ]
        client.upsert(collection_name=name, points=points, wait=True)
        total += len(points)
    return total

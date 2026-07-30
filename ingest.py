#!/usr/bin/env python
"""Main runner: fetch -> parse -> chunk -> embed -> upsert, for every filing.

    python ingest.py                      # all companies, all years
    python ingest.py --tickers AAPL MSFT  # a subset
    python ingest.py --years 2024
    python ingest.py --dry-run            # parse and chunk, write nothing

A filing that EDGAR does not have, or that fails to parse, is logged and skipped
-- one bad filing never aborts the run.
"""

import argparse
import logging
import sys
import time

import config
from ingest.chunk import chunk_text
from ingest.embed import Embedder
from ingest.fetch import find_filings, init_edgar
from ingest.parse import extract_sections
from ingest.upsert import ensure_collection, get_client, upsert_chunks

log = logging.getLogger("ingest")


def build_records(filing_ref, sections, embedder, chunk_size, overlap):
    """Chunk every section of one filing and return upsert-ready records.

    Chunking is per-section, so no chunk ever spans two sections.
    """
    pending = []  # (payload-without-vector, text)
    for section in sections:
        chunks = chunk_text(
            section.text,
            count_tokens=embedder.count_tokens,
            encode_tokens=embedder.encode_tokens,
            decode_tokens=embedder.decode_tokens,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        if not chunks:
            continue
        for index, text in enumerate(chunks):
            pending.append({
                "text": text,
                "ticker": filing_ref.ticker,
                "company": filing_ref.company,
                "year": filing_ref.year,
                "part": section.part,
                "section": section.item,
                "section_label": section.label,
                "chunk_index": index,
                "total_chunks_in_section": len(chunks),
                "is_sparse": section.is_sparse,
                "incorporated_by_reference": section.incorporated_by_reference,
                "filing_id": filing_ref.filing_id,
                "period_of_report": filing_ref.period_of_report,
            })

    vectors = embedder.embed([p["text"] for p in pending])
    return [{"payload": p, "vector": v} for p, v in zip(pending, vectors)]


def process_filing(filing_ref, embedder, client, args, chunk_size):
    """Ingest a single filing. Returns a summary dict, or None if it failed."""
    tag = f"[{filing_ref.ticker} {filing_ref.year}]"
    started = time.time()

    try:
        sections, missing = extract_sections(filing_ref)
    except Exception as exc:
        log.error("%s failed to parse filing %s: %s -- skipping",
                  tag, filing_ref.filing_id, exc)
        return None

    if not sections:
        log.error("%s no sections extracted from %s -- skipping",
                  tag, filing_ref.filing_id)
        return None

    sparse = [s.item for s in sections if s.is_sparse]
    by_ref = [s.item for s in sections if s.incorporated_by_reference]

    log.info("%s filing %s period %s | %d sections: %s",
             tag, filing_ref.filing_id, filing_ref.period_of_report,
             len(sections), ", ".join(s.item for s in sections))
    if sparse:
        log.info("%s   sparse: %s", tag, ", ".join(sparse))
    if by_ref:
        log.info("%s   incorporated by reference: %s", tag, ", ".join(by_ref))
    if missing:
        log.warning("%s   sections absent from filing: %s", tag, ", ".join(missing))

    records = build_records(filing_ref, sections, embedder, chunk_size,
                            config.CHUNK_OVERLAP_TOKENS)
    log.info("%s   chunks created: %d", tag, len(records))

    if args.dry_run:
        log.info("%s   dry-run -- nothing upserted (%.1fs)", tag, time.time() - started)
        upserted = 0
    else:
        upserted = upsert_chunks(client, config.COLLECTION_NAME, records,
                                 config.UPSERT_BATCH_SIZE)
        log.info("%s   chunks upserted: %d (%.1fs)", tag, upserted,
                 time.time() - started)

    return {
        "ticker": filing_ref.ticker,
        "year": filing_ref.year,
        "filing_id": filing_ref.filing_id,
        "sections": len(sections),
        "missing": missing,
        "sparse": len(sparse),
        "by_ref": len(by_ref),
        "chunks": len(records),
        "upserted": upserted,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Ingest SEC 10-K filings into Qdrant.")
    parser.add_argument("--tickers", nargs="+", default=config.COMPANIES)
    parser.add_argument("--years", nargs="+", type=int, default=config.YEARS)
    parser.add_argument("--dry-run", action="store_true",
                        help="parse, chunk and embed but do not write to Qdrant")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    init_edgar(config.SEC_IDENTITY)
    embedder = Embedder(config.EMBEDDING_MODEL, config.EMBEDDING_BATCH_SIZE)

    if embedder.dim != config.EMBEDDING_DIM:
        log.warning("config.EMBEDDING_DIM=%d but %s produces %d -- using %d",
                    config.EMBEDDING_DIM, config.EMBEDDING_MODEL,
                    embedder.dim, embedder.dim)

    # Reserve room for the model's [CLS]/[SEP] tokens so no chunk is truncated
    # at encode time.
    chunk_size = min(config.CHUNK_SIZE_TOKENS, embedder.max_seq_length - 2)
    if chunk_size < config.CHUNK_SIZE_TOKENS:
        log.warning("chunk size reduced from %d to %d to fit %s (max_seq_length=%d)",
                    config.CHUNK_SIZE_TOKENS, chunk_size, config.EMBEDDING_MODEL,
                    embedder.max_seq_length)
    log.info("model=%s dim=%d chunk=%d overlap=%d",
             embedder.model_name, embedder.dim, chunk_size,
             config.CHUNK_OVERLAP_TOKENS)

    client = None
    if not args.dry_run:
        client = get_client(config.QDRANT_URL)
        ensure_collection(client, config.COLLECTION_NAME, embedder.dim)

    summaries, failures = [], []
    for ticker in args.tickers:
        try:
            refs = find_filings(ticker, args.years)
        except Exception as exc:
            log.error("[%s] could not list filings: %s -- skipping company", ticker, exc)
            failures.append((ticker, None, str(exc)))
            continue

        for ref in refs:
            summary = process_filing(ref, embedder, client, args, chunk_size)
            if summary is None:
                failures.append((ticker, ref.year, "parse/ingest failed"))
            else:
                summaries.append(summary)

    log.info("=" * 72)
    log.info("%-7s %-6s %-22s %8s %8s %8s", "TICKER", "YEAR", "FILING", "SECTIONS",
             "CHUNKS", "UPSERTED")
    for s in summaries:
        log.info("%-7s %-6d %-22s %8d %8d %8d", s["ticker"], s["year"],
                 s["filing_id"], s["sections"], s["chunks"], s["upserted"])
    log.info("-" * 72)
    log.info("%d filings ingested, %d chunks, %d upserted",
             len(summaries), sum(s["chunks"] for s in summaries),
             sum(s["upserted"] for s in summaries))
    if failures:
        log.warning("%d failures: %s", len(failures),
                    "; ".join(f"{t} {y or ''} ({why})" for t, y, why in failures))

    if not args.dry_run and client is not None:
        count = client.count(config.COLLECTION_NAME, exact=True).count
        log.info("collection '%s' now holds %d points", config.COLLECTION_NAME, count)

    return 0 if summaries else 1


if __name__ == "__main__":
    sys.exit(main())

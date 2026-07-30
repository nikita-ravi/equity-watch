"""Phoenix / OpenTelemetry tracing for the retrieval layer.

Every search() call becomes one RETRIEVER span carrying the query, the filters
applied, latency, and every returned chunk with its score and metadata -- so two
queries can be compared side by side in the Phoenix UI.

Tracing is optional and never fatal: if Phoenix is not installed or not
reachable, retrieval still runs and the spans are dropped.
"""

import json
import logging

import config

log = logging.getLogger(__name__)

_tracer = None
_session = None
_enabled = False


def start_phoenix_ui():
    """Launch the local Phoenix app and return its URL, or None on failure.

    If something is already serving on the Phoenix port -- an earlier CLI run, or
    `phoenix serve` -- that instance is left alone and reused.
    """
    global _session
    if _session is not None:
        return _session.url

    if _port_in_use(config.PHOENIX_PORT):
        url = f"http://localhost:{config.PHOENIX_PORT}"
        log.info("reusing Phoenix already running at %s", url)
        return url

    try:
        import phoenix as px
        _session = px.launch_app(port=config.PHOENIX_PORT)
        return _session.url
    except Exception as exc:
        log.warning("could not launch Phoenix UI: %s", exc)
        return None


def init_tracing():
    """Point OpenTelemetry at Phoenix. Safe to call more than once."""
    global _tracer, _enabled
    if _tracer is not None:
        return _tracer

    if not config.PHOENIX_ENABLED:
        log.debug("Phoenix tracing disabled via PHOENIX_ENABLED")
        _tracer = _NoopTracer()
        return _tracer

    try:
        from phoenix.otel import register
        provider = register(
            project_name=config.PHOENIX_PROJECT,
            endpoint=config.PHOENIX_ENDPOINT,
            batch=False,          # CLI processes are short-lived; export eagerly
            set_global_tracer_provider=False,
            verbose=False,
        )
        _tracer = provider.get_tracer(__name__)
        # Kept so flush() can reach the provider on process exit.
        _tracer._real_provider = provider
        _enabled = True
        log.info("tracing to Phoenix project '%s'", config.PHOENIX_PROJECT)
    except Exception as exc:
        log.warning("tracing disabled (%s)", exc)
        _tracer = _NoopTracer()
    return _tracer


def get_tracer():
    return _tracer if _tracer is not None else init_tracing()


def flush():
    """Force-export pending spans before the process exits."""
    if not _enabled:
        return
    provider = getattr(_tracer, "_real_provider", None)
    if provider is not None and hasattr(provider, "force_flush"):
        try:
            provider.force_flush()
        except Exception as exc:
            log.debug("span flush failed: %s", exc)


def record_results(span, results, max_chars=None):
    """Attach retrieved chunks to a span using OpenInference conventions.

    Phoenix renders these as an inspectable document list on the span.
    """
    if span is None or not hasattr(span, "set_attribute"):
        return
    try:
        from openinference.semconv.trace import SpanAttributes
    except Exception:
        return

    prefix = SpanAttributes.RETRIEVAL_DOCUMENTS
    for i, r in enumerate(results):
        text = r.text if max_chars is None else r.text[:max_chars]
        base = f"{prefix}.{i}.document"
        span.set_attribute(f"{base}.id", _point_ref(r))
        span.set_attribute(f"{base}.content", text)
        span.set_attribute(f"{base}.score", float(r.score))
        # Metadata must be a JSON string; OTel attributes cannot hold dicts.
        span.set_attribute(f"{base}.metadata", json.dumps({
            "ticker": r.ticker,
            "company": r.company,
            "year": r.year,
            "part": r.part,
            "section": r.section,
            "section_label": r.section_label,
            "chunk_index": r.chunk_index,
            "total_chunks_in_section": r.metadata.get("total_chunks_in_section"),
            "is_sparse": r.metadata.get("is_sparse"),
            "incorporated_by_reference": r.metadata.get("incorporated_by_reference"),
            "filing_id": r.filing_id,
            "period_of_report": r.metadata.get("period_of_report"),
        }))


def _point_ref(result):
    return f"{result.filing_id}|{result.section}|{result.chunk_index}"


def _port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        return sock.connect_ex(("127.0.0.1", port)) == 0


class _NoopSpan:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def set_attribute(self, *_a, **_kw):
        pass

    def set_status(self, *_a, **_kw):
        pass

    def record_exception(self, *_a, **_kw):
        pass


class _NoopTracer:
    def start_as_current_span(self, *_a, **_kw):
        return _NoopSpan()

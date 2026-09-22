# sec-rag

Retrieval over SEC Form 10-K filings — 10 US large-caps, fiscal years 2022–2024,
30 filings, ~9,800 chunks. Four retrieval versions, each measured against a
hand-built ground-truth set, plus a ReAct agent that answers with citations
traceable to an EDGAR accession number.

The retrieval layer is the project. The agent was built last and deliberately
changed nothing underneath it.

```bash
# retrieval only, no LLM
python retrieve.py "What are Apple's main risk factors?" --ticker AAPL --year 2024 --retrieval v4

# the agent
python -m agent.run_agent "How did Apple's revenue change from FY2022 to FY2024?"
```

---

## Contents

- [Status](#status)
- [How it works](#how-it-works)
- [Results](#results)
- [Design decisions](#design-decisions)
- [Failure modes](#failure-modes)
- [Development log](#development-log)
- [Evaluation: what it measures and what it does not](#evaluation-what-it-measures-and-what-it-does-not)
- [Setup](#setup)
- [Layout](#layout)

---

## Status

| Component | State |
|---|---|
| Ingest (EDGAR → sections → chunks → Qdrant) | Working |
| Retrieval v1 dense | Working, measured |
| Retrieval v2 hybrid (BM25 + RRF) | Working, measured |
| Retrieval v3 structured query parsing | Working, measured |
| Retrieval v4 cross-encoder rerank | Working, measured |
| Section quota on the candidate pool | Implemented, **not yet measured end to end** |
| XBRL reported financials + reconciliation gate | Working, measured on 240 extractions |
| ReAct agent (3 tools) | Working, **never measured by the eval** |
| Answer-faithfulness eval | **Not built** — the largest gap |

**Two things must happen before the retrieval numbers below can be trusted again:**

1. **Re-ingest is required and needs a delete step first.** The parser fix
   (2026-09-13) moves content between sections. Point IDs are
   `uuid5(filing_id|section|chunk_index)`, so moved content writes *new* points
   and the stale ones survive — `ingest.py` has no delete path. Drop the
   collection or delete by `filing_id` before re-ingesting, or the corrected
   corpus will be worse than the broken one.
2. **The stored eval results are stale.** Every file in `eval/results/` reports
   `num_questions=50`; the dataset is now 65. They are not reproducible as-is.

---

## How it works

### Ingest

```
EDGAR (edgartools) → section extraction → token-aware chunking → bge embed → Qdrant
```

Filings are located by `period_of_report`, not filing date, so "fiscal year" is
comparable across filers with different fiscal calendars. Amendments (10-K/A)
are excluded — they carry only Part III.

Each of the 23 named Item sections is extracted separately and chunked to 512
tokens with 64 overlap, **measured in the embedding model's own tokenizer**, so
a chunk is exactly what the model encodes. Chunking is scoped per section, so a
chunk can never straddle a section boundary — which is what makes the `section`
field trustworthy as a filter.

Two flags are set at ingest and used as exclusions at query time:

- `is_sparse` — a section under 200 characters, or a null answer ("None.")
- `incorporated_by_reference` — a pointer to an exhibit rather than disclosure

### Retrieval

| Version | What it adds |
|---|---|
| **v1** | Dense cosine over Qdrant, with optional metadata filters |
| **v2** | BM25 in-process, fused with dense via Reciprocal Rank Fusion (k=60) |
| **v3** | Parse ticker / year / section out of the question, then run v2 |
| **v4** | Widen the pool, budget it per section, re-score with a cross-encoder |

Each version is a superset of the previous and is selectable by flag, so every
baseline stays reproducible.

### Agent

A single ReAct agent (llama-3.3-70b on Groq) with three tools:

- `search_10k` — retrieval, scoped to **one company and one fiscal year**
- `list_corpus` — what the corpus actually contains
- `get_financials` — exact reported figures from XBRL

The single-filing scope is deliberate: it forces decomposition to be explicit.
A three-year trend question becomes three calls, and the model compares the
figures itself.

---

## Results

Stored runs, top_k=5, unfiltered unless noted. **These are the n=50 runs** —
see [Status](#status).

| Version | keyword_hit | perfect | zero | section_hit | latency |
|---|---|---|---|---|---|
| v1 dense | 0.780 | 0.42 | 0.02 | 0.84 | 95 ms |
| v2 +BM25 RRF | 0.845 | 0.50 | 0.00 | 0.86 | 171 ms |
| v3 +query parsing | 0.868 | 0.62 | 0.00 | 0.84 | 116 ms |
| v4 +rerank | 0.875 | 0.58 | 0.00 | 0.88 | 7,153 ms |

The real gain is v1 → v3: keyword hit 78% → 87%, and the rate at which *every*
expected keyword surfaced 42% → 62%. v4 buys ~4 points of section accuracy for
a 60× latency cost.

**`top1_score` is excluded from this table on purpose.** It is recorded in the
result files but is not comparable across versions: `rerank()` overwrites
`result.score`, so the field is cosine similarity in v1 (~0.78), an RRF score in
v2/v3 (~0.03), and a cross-encoder probability in v4 (~0.94). Read left to
right it looks like a 29× improvement; it is a change of units.

### XBRL financials

Across 240 extractions (30 filings × 8 metrics): **204 usable, 0 unreconciled,
36 honest gaps**, and 9/9 exact matches against the eval set's expected answers
— including the two the library's own getters get wrong.

---

## Design decisions

**Fiscal year from `period_of_report`.** Apple's FY2024 10-K covers a period
ending September 2024 but was filed in November. Keying off the period is what
makes the year comparable across filers. A second-order fix sits on top:
`FISCAL_YEAR_OFFSET = {"HD": 1}`, because Home Depot names a fiscal year after
its *starting* calendar year while Walmart names it after the ending one.

**`bge-small-en-v1.5` over `all-MiniLM-L6-v2`.** Both are 384-dim, but bge has
a 512-token limit against MiniLM's 256. With 512-token chunks, MiniLM would
silently truncate half of every chunk. `EMBEDDING_DIM` is verified against the
loaded model at runtime so a model swap cannot quietly corrupt the index.

**Chunk in the model's own tokens.** The chunker is handed the embedder's
tokenizer, so a chunk can never overflow what the model encodes. It packs whole
sentences, with an abbreviation list so it does not shred `U.S. federal law`.
A flattened financial table is often one giant unpunctuated "sentence", so
anything too large is hard-split on overlapping token windows.

**BM25's tokenizer keeps numerals intact.** `383,285` and `$1.5` survive as
single tokens. A default word tokenizer shreds them into `383` and `285` —
destroying exactly the lexical signal BM25 was added to provide.

**IDF stays corpus-wide.** BM25 scores the whole corpus, *then* filters, then
cuts to top-n. Filtering first would compute IDF over a handful of documents.

**RRF rather than score blending.** BM25 scores are unbounded, cosine sits in
[-1, 1]; they cannot be meaningfully summed. RRF fuses on rank alone.

The BM25 weight has a derivable breaking point. With `k=60` and 50 candidates,
the worst dual-list score is `(1+w)/110` and the best dense-only score is
`1/61`. They cross at `w = 110/61 − 1 = 0.8033`. Below that, a chunk found by
*both* retrievers can rank below a chunk found by dense alone — which inverts
the point of fusion. Default is 1.0.

**Query parsing is rules-based, not an LLM call.** The entity space is closed:
10 companies, 3 years, 23 sections. Rules are deterministic, free, zero-latency
and debuggable. This does not generalise — at 500 companies the alias table
becomes unmaintainable and an entity-linking step would be the right answer.

**Section filtering defaults to OFF.** Measured on this corpus, most
answer-bearing chunks are *not* in the section the question nominally targets
(revenue figures sit in Item 8 tables even for Item 7 questions), so a hard
filter removed more right answers than wrong ones. `parse_query` still returns
the section for callers that want it.

> **Known inconsistency.** The agent's system prompt actively instructs the
> model to pass `section=`, and explicit arguments bypass `V3_SECTION_FILTER`.
> So the agent routinely does the thing the eval concluded is net-harmful. The
> two decisions were made at different times and never reconciled, and the eval
> never caught it because it passes `section=None` and never exercises the agent
> path. Unresolved.

**`must_not` rather than matching `False`.** Excluding `is_sparse` uses
`must_not(is_sparse=True)`, so a chunk written before the flag existed stays
retrievable.

**Deterministic point IDs.** `uuid5(filing_id|section|chunk_index)`. Section is
part of the key because `chunk_index` is per-section and would otherwise collide.
See the caveat in [Status](#status).

**The reranker receives the original question, never `clean_query`.**
Cross-encoders are trained on natural-language pairs; stripping entities would
deprive the reranker of the same context it deprived the retriever of.

**`AGENT_TOP_K` is not exposed to the model.** It has no basis for choosing a
value, and every extra tool parameter is one more thing a model can emit
malformed.

**Validation degrades, never raises.** An LLM will confidently pass
`ticker="APPL"` or `year=2025`. A hard failure ends the run, so the validator
fuzzy-matches, clamps, or drops the filter — and records every adjustment,
which is surfaced to the model in the tool output.

**Statement-level XBRL, not `companyfacts`.** `companyfacts` returns every
tagged fact including segment and product breakdowns with no flag marking the
consolidated total. Selecting naively returned **$43,715M** for Apple's FY2023
total assets against a real **$352,583M**, $663M for Pfizer, and $21.4B for J&J
revenue. Every one looked like an ordinary number.

**Nothing is trusted on a getter's word.** Every extracted figure is reconciled
against the statement line it should have come from. A figure that does not
reconcile is stored as **missing, never as a number** — a gap is visible and
gets fixed; a plausible wrong number is neither.

---

## Failure modes

| # | Symptom | Root cause | Status |
|---|---|---|---|
| FM-1 | Table chunks lose to mediocre prose | Tables carry no BM25 signal; a dense-only chunk scores `1/61` and loses to any dual-list chunk | Fixed (v4 cross-encoder) |
| FM-2 | Item 1A crowded out of results | Structural volume — JNJ has 145 Item 8 chunks to 19 Item 1A. A flat top-20 is 20 Item 8 and zero Item 1A | Fixed (section quota), **unmeasured** |
| FM-3 | Sparse sections unreachable / noisy | "None." sections indistinguishable from real ones | Fixed (`is_sparse` + `must_not`) |
| FM-4 | XOM MD&A questions return Item 16 | XOM answers Item 7/8 with "Reference is made to…" and appends the Financial Section after Item 16; the detector matched only "incorporated by reference" | Fixed in parser, **needs re-ingest** |
| FM-5 | WMT FY2024 content indexed twice | Item 4 ran on for 87,852 chars, swallowing Part II; boundary detection missed `ITEM 5.MARKET` (no space) | Fixed in parser, **needs re-ingest** |
| FM-6 | Confidently wrong financial figures | `companyfacts` dimensional members; then the library's own getters picking neighbouring lines | Fixed (reconciliation gate) |
| FM-7 | Stale chunks survive re-ingest | Point ID includes section; moved content writes new points and no delete path exists | **Open** |
| FM-8 | `top1_score` incomparable across versions | `rerank()` overwrites `result.score`; eval records it blind | **Open** |
| FM-9 | Stored eval results unreproducible | 15 questions added after the runs; all files say `n=50` | **Open** |
| FM-10 | Agent pins sections the eval says are harmful | Two decisions made at different times; eval never exercises the agent path | **Open** |
| FM-11 | Groq emits malformed tool calls | Generation fluke — `<function=…>` as text, rejected 400 | Mitigated (3 retries + fallback model) |
| FM-12 | Long chunks truncated at rerank | A 2,200-char chunk exceeds 512 tokens; the tail is never scored | **Open**, documented |

---

## Development log

Reconstructed from file creation times; commit dates match.

### 2026-07-29 — Ingest

EDGAR fetch, section extraction, token-aware chunking, Qdrant upsert. 30
filings across 10 companies and 3 fiscal years.

Parse bugs found and fixed during this phase:

- **Page-footer bleed.** `Apple Inc. | 2022 Form 10-K | 19` bleeding into
  whichever section straddled a page break, making null answers look
  substantive. Stripped by regex.
- **Signature block appended to the last Item.** Everything from `SIGNATURES`
  onward belongs to no item; now truncated.
- **Heading stripper eating the body.** EDGAR sometimes runs a section title
  straight into the first sentence
  (`Executive CompensationThe information required by...`), so a stripper that
  assumed the heading was its own line consumed the whole section. Now only the
  item *number* is removed.

### 2026-07-29 — Retrieval v1

Dense cosine with optional metadata filters, Phoenix/OpenTelemetry tracing, and
a CLI. Baseline **78.0% keyword_hit@5** unfiltered.

The anomaly that set the agenda: *easy* questions scored **below** medium ones.
Easy questions are exact-numeral lookups, which is dense retrieval's weakest
case — a table of bare figures carries no semantic signal for a cosine model.

### 2026-07-30 — Retrieval v2

BM25 + RRF, targeting exactly that weakness. **78.0% → 84.5%**, zero-hit rate
to 0.00.

### 2026-07-31 — Retrieval v3

Structured query parsing. **84.5% → 86.8%**, and perfect-recall 50% → 62% —
the single largest accuracy gain in the project, from *not* using a model.

### 2026-07-31 — Retrieval v4

Cross-encoder reranking over a widened candidate pool. **86.8% → 87.5%**
keyword hit, section hit 0.84 → 0.88, latency 116 ms → 7,153 ms.

A shallower pool (10 instead of 20) was tried as a latency optimisation and
**reverted**: it was justified by two eval questions rescued from RRF ranks ≤10,
but it does not generalise — RRF buries Item 1A below rank 10 whenever Item 7/8
crowd the pool, so risk-factor questions lost their Item 1A evidence entirely.
The reason is recorded in `config.py` so it is not retried.

### 2026-07-31 — Agent

ReAct loop, argument validation, relevance floor, LangSmith tracing. The agent
is a front end over a retrieval stack that was already finished and measured:
`search()`'s signature is unchanged and `retrieve.py` still works.

Phoenix tracing is disabled on the agent path — its export is synchronous, and
inside LangGraph the tool runs on a worker thread while the main thread waits,
so a blocked export deadlocks both. `TORCH_DEVICE` is forced to `cpu` for the
same reason: MPS plus LangGraph's threads/forks segfaults the interpreter.

### 2026-08-21 — Eval expansion, 50 → 65

The original 50-question set had a structural bias: **33 of 50 targeted Item 7**
(MD&A financial figures), which is what the system is already best at. Aggregate
keyword_hit was saturated and blind to the conceptual sections.

15 section-targeted questions were added, including two **intentional canaries**
on Item 1B — a section whose entire content is "None.", flagged `is_sparse` and
excluded from retrieval. Those questions must score 0% `section_hit` while the
sparse filter is on; they exist to keep that policy visible.

Current composition: Item 7 ×33, Item 1 ×14, Item 1A ×11, Item 1C ×3,
Item 1B ×2, Item 7A ×2. By type: factual 30, numerical 21, comparative 11,
trend 3.

### 2026-09-13 — Parser fixes, section quota, XBRL financials

Three changes, in the order the evidence arrived. See
[Failure modes](#failure-modes) FM-4, FM-5, FM-2, FM-6 and the commit messages
for the measurements.

The parser bug was found by reading per-question eval output — every chunk
returned for XOM's MD&A questions was labelled `Item 16`. Tracing did not find
it; the eval did, and had been flagging it for weeks as low section accuracy
that was read as a ranking problem.

---

## Evaluation: what it measures and what it does not

65 questions across all 10 companies, hand-labelled with expected answer
keywords and tagged by type and difficulty.

**Primary metric — `keyword_hit@k`:** of a question's expected keywords, what
fraction appear anywhere in the top-k chunk texts. Deliberately generous: it
asks *"did retrieval put the answer in front of the model"*, not *"did the model
answer correctly"* — the right question for a layer with no LLM in it.

**Supporting metrics** separate failure kinds: `ticker_hit` and `year_hit`
(right document?) from `section_hit` (right part of it?). `section_hit` is the
discriminating one for failure discovery.

### Known weaknesses

- **No answer-faithfulness measurement at all.** The eval proves the right
  chunks came back. It cannot prove the answer reflects them. This is the
  largest gap in the project.
- **n=65 is small.** 95% confidence intervals are roughly ±10pp — wider than
  several of the deltas above. v1→v3 is probably real; v3→v4 probably is not.
- **The agent path is never exercised.** `eval/` calls `search()` directly, so
  every number describes the pre-agent architecture. This is why FM-10 survived
  unnoticed.
- **Ground truth is unverified.** The dataset header admits keywords come from
  the author's reading, with `# ⚠ verify` markers. At least one confirmed error:
  `XOM-2023-7-002` expects `26,279`, which appears nowhere in the filing — XOM
  writes "26.3 billion".
- **Keyword matching is substring-based.** `"36,010" in haystack` also matches
  inside `"136,010"`. No word-boundary check.
- **No keyword weighting.** Missing `"volatility"` counts the same as missing
  `"36,010"`.
- **Two result files measure nothing.** `v1_*.prefix.json` claim to test the bge
  query-instruction prefix. The two runs are retrieval-identical — same chunks,
  same order, same scores, all 50 questions. The only differences come from the
  ground-truth keywords being edited between runs. They should be deleted or
  re-run.

### Next

1. Delete path in `ingest.py`, then re-ingest (FM-7 blocks everything else).
2. Re-run all configs at n=65 so the numbers mean something again.
3. Answer-faithfulness eval: invented number / unsupported superlative / wrong
   ordering / number attributed to the wrong company — with "derived from two
   retrieved figures" explicitly carved out as *not* a hallucination. The number
   check should be code, not an LLM judge; self-critique fails precisely on
   factual errors.
4. Route the reranker instead of always running it — skip v4 when ticker, year
   and section are already pinned.

---

## Setup

Python 3.11. Qdrant on `localhost:6333` (Docker).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

docker run -p 6333:6333 -v "$(pwd)/../qdrant_storage:/qdrant/storage" qdrant/qdrant

cp .env.example .env     # add GROQ_API_KEY and LANGCHAIN_API_KEY
python ingest.py         # ~30 filings; see the FM-7 caveat before re-running
```

`.env` is gitignored. `.env.example` documents the required keys.

```bash
python eval/run_eval.py --retrieval v3 --output eval/results/v3_unfiltered.json
python eval/compare.py eval/results/v1_unfiltered.json eval/results/v3_unfiltered.json
```

---

## Layout

```
config.py                 all tuning constants, with the reasoning for each
ingest.py                 fetch → parse → chunk → embed → upsert
retrieve.py               retrieval CLI (no LLM)

ingest/
  fetch.py                EDGAR via edgartools, keyed on period_of_report
  parse.py                section extraction, cross-reference + boundary repair
  chunk.py                sentence-aware, tokenizer-exact chunking
  embed.py                model wrapper; also exposes the tokenizer
  upsert.py               Qdrant collection management, deterministic point IDs
  financials.py           XBRL figures behind the reconciliation gate

retrieve/
  search.py               entry point and v1 dense
  bm25.py                 in-process lexical index, numeral-preserving tokenizer
  fusion.py               Reciprocal Rank Fusion
  hybrid.py               v2
  query_parser.py         ticker / year / section from natural language
  search_v3.py            v3
  quota.py                per-section budgeting of the candidate pool
  rerank.py               cross-encoder
  search_v4.py            v4
  result.py               SearchResult
  tracing.py              Phoenix / OpenTelemetry spans

agent/
  agent.py                ReAct loop, system prompt, retry + fallback
  tools.py                search_10k, list_corpus, get_financials
  validation.py           coerce LLM arguments; degrade, never raise
  run_agent.py            CLI

eval/
  dataset.py              65 ground-truth questions
  retrieval_eval.py       metric computation
  run_eval.py             runner
  compare.py              diff two result files
  results/                stored runs (n=50 — stale, see Status)

verification_report.md    manual chunk-level dump for 4 questions, one-off
```

---

## A note on scope

This is a research and measurement project, not a product. It is deliberately
small — 10 companies, 3 years, one corpus — because the goal was to make each
retrieval decision measurable rather than to scale. Where a decision was made on
intuition rather than measurement, the README says so.

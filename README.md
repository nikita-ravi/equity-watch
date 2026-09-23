# Equity Watch

Ask plain-English questions about SEC 10-K filings and get answers grounded in
the filings themselves — every figure traceable to an EDGAR accession number.

Four retrieval versions, each measured against a hand-built ground-truth set.

---

## What It Does

You ask *"How did Apple's revenue change from FY2022 to FY2024?"*. A ReAct agent
splits that into one retrieval per fiscal year, each scoped to a single filing.
Retrieval runs dense search and BM25 in parallel, fuses them by rank, budgets
the candidate pool so one section can't crowd out the others, and re-scores the
survivors with a cross-encoder. Anything below a relevance floor is dropped
*before* the model sees it. The agent answers only from what came back, and
cites it.

Exact financial figures don't go through retrieval at all — they're read from
the filing's own XBRL data and reconciled against the statement they came from.

**Corpus:** 10 US large-caps × FY2022–2024 = 30 filings, ~9,800 chunks.

---

## Architecture

```
┌──────────────────────────────────────────────┐
│  retrieve.py            agent/run_agent.py   │
│  (retrieval CLI)        (agent CLI)          │
└───────────┬──────────────────┬───────────────┘
            │                  │
            │     ┌────────────▼───────────────┐
            │     │   Guardrails · input       │
            │     │   scope · advice           │
            │     └────────────┬───────────────┘
            │     ┌────────────▼───────────────┐
            │     │   ReAct Agent              │ ◄── LangSmith
            │     │   llama-3.3-70b · Groq     │     (agent traces)
            │     │                            │
            │     │   search_10k               │
            │     │   list_corpus              │
            │     │   get_financials ──────────┼──┐
            │     └────────────┬───────────────┘  │
            │     ┌────────────▼───────────────┐  │
            │     │   Guardrails · output      │  │
            │     │   every figure · every cite│  │
            │     └────────────────────────────┘  │
            │                  │                  │
┌───────────▼──────────────────▼───────────────┐  │
│   Retrieval                                  │  │
│                                              │  │
│   v1  dense (cosine)                         │  │ ◄── Phoenix
│   v2  + BM25, fused by RRF                   │  │  (retrieval
│   v3  + ticker/year/section parsed from text │  │   traces)
│   v4  + section quota, cross-encoder rerank  │  │
│       + relevance floor 0.3                  │  │
└───────────┬──────────────────────────────────┘  │
            │                                     │
┌───────────▼──────────────┐    ┌─────────────────▼─────────────┐
│   Qdrant :6333           │    │   XBRL financials             │
│   sec_10k · ~9,800 chunks│    │   statement API + reconcile   │
│   384-dim cosine         │    │   (bypasses retrieval)        │
└───────────▲──────────────┘    └─────────────────▲─────────────┘
            │                                     │
┌───────────┴─────────────────────────────────────┴─────────────┐
│   Ingest:  EDGAR (edgartools)                                 │
│            → 23 Item sections → 512-token chunks → bge-small  │
└───────────────────────────────────────────────────────────────┘
```

Each retrieval version is a superset of the previous one and selectable by flag,
so every baseline stays reproducible.

---

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

docker run -p 6333:6333 -v "$(pwd)/../qdrant_storage:/qdrant/storage" qdrant/qdrant

cp .env.example .env        # SEC_IDENTITY and GROQ_API_KEY are required
python ingest.py            # ~30 filings
```

```bash
# retrieval only, no LLM
python retrieve.py "What are Apple's main risk factors?" --ticker AAPL --year 2024 --retrieval v4

# the agent
python -m agent.run_agent "How did Apple's revenue change from FY2022 to FY2024?"

# evaluation
python eval/run_eval.py --retrieval v3 --output eval/results/v3_unfiltered.json
python eval/compare.py eval/results/v1_unfiltered.json eval/results/v3_unfiltered.json
```

---

## Project Structure

| Path | What it is |
|---|---|
| `ingest/` | EDGAR fetch, section extraction, chunking, embedding, Qdrant upsert, XBRL financials |
| `retrieve/` | The four retrieval versions, BM25, RRF fusion, query parser, section quota, reranker |
| `agent/` | ReAct loop, the three tools, LLM argument validation |
| `guardrails/` | Deterministic input and output checks, plus their tests |
| `eval/` | 65 ground-truth questions, retrieval metrics, the faithfulness harness, stored results |
| `config.py` | Every tuning constant, with the reasoning for each in a comment |
| `retrieve.py` | Retrieval CLI — no LLM involved |

---

## Results

Stored runs, `top_k=5`, unfiltered.

| Version | keyword_hit | all keywords found | section_hit | latency |
|---|---|---|---|---|
| v1 dense | 0.780 | 0.42 | 0.84 | 95 ms |
| v2 + BM25 / RRF | 0.845 | 0.50 | 0.86 | 171 ms |
| v3 + query parsing | 0.868 | 0.62 | 0.84 | 116 ms |
| v4 + rerank | 0.875 | 0.58 | 0.88 | 7,153 ms |

The real gain is v1 → v3: keyword hit 78% → 87%, and the rate at which *every*
expected keyword surfaced 42% → 62%. v4 buys ~4 points of section accuracy for
a 60× latency cost.

**XBRL financials:** across 240 extractions (30 filings × 8 metrics) —
204 usable, 0 unreconciled, 36 honest gaps, and 9/9 exact matches against the
eval set's expected answers.

---

## Key Engineering Decisions

**Why BM25 alongside dense retrieval?** Dense embeddings encode meaning, not
tokens. A question asking for `383,285` gets no help from a model that treats
numerals as low-signal noise, and financial questions are full of them. BM25
matches the token exactly. The detail that makes it work is the tokenizer: it
deliberately keeps `383,285` and `$1.5` intact, where a default word tokenizer
shreds them into `383` and `285` — destroying the signal BM25 was added to
provide. Worth 6.5 points of keyword hit.

**Why RRF instead of blending the scores?** BM25 scores are unbounded; cosine
sits in [-1, 1]. Adding or averaging them is meaningless. Reciprocal Rank Fusion
merges on rank position alone, so neither scale dominates. The BM25 weight has a
derivable breaking point: with `k=60` and 50 candidates, the worst dual-list
score is `(1+w)/110` and the best dense-only score is `1/61`; they cross at
`w = 110/61 − 1 = 0.8033`. Below that, a chunk found by *both* retrievers can
rank below one found by dense alone — which inverts the point of fusion.

**Why parse the query with rules instead of an LLM?** The entity space is
closed: 10 companies, 3 years, 23 sections. Every surface form someone might
type — `alphabet`, `google`, `alphabet inc` — can simply be written down. Rules
are free, deterministic, zero-latency and debuggable; an LLM would be *guessing
at* something that can be *enumerated*. This produced the single largest
accuracy jump in the project, from not using a model. It does not generalise —
at 500 companies the alias table becomes unmaintainable.

**Why budget the candidate pool per section?** The imbalance is structural, not
incidental: JNJ FY2023 has 145 Item 8 chunks to 19 Item 1A. A flat top-20 fills
with Item 8 on volume before relevance is considered at all, so a risk-factors
question can end up with zero Item 1A chunks and nothing for the reranker to
rescue. So the draw is widened, then any one section is capped at half the pool.
The cap reallocates slots rather than discarding them.

**Why a cross-encoder on top of fusion?** A bi-encoder embeds query and chunk
independently and can never model how they interact — which is why a table of
bare numbers scores poorly against a question that never names those numbers. A
cross-encoder reads the pair jointly. It's far too slow for the corpus, so it
only re-scores 20 candidates. It receives the *original* question, never the
entity-stripped one, because it's trained on natural-language pairs.

**Why read financial figures from XBRL instead of retrieving them?** A retrieved
figure has to survive being flattened out of a table into a chunk, be retrieved,
and then be read correctly off that flattened text. The filing already carries
the number as typed data. This covers 9 of the 21 numerical eval questions; the
rest are segment-level, narrative counts, or bank-specific lines, so it's a
complement to retrieval, not a replacement.

**Why not the `companyfacts` API?** It returns every tagged fact including
segment and product breakdowns, with no flag marking the consolidated total.
Selecting naively returned **$43,715M** for Apple's FY2023 total assets against
a real **$352,583M** — and $663M for Pfizer, $21.4B for J&J revenue. Every one
looked like a perfectly ordinary number. The statement-level API is already
dimension-filtered and period-scoped.

**Why reconcile every extracted figure?** Because the library's own normalised
getters are quietly wrong some of the time. `get_total_liabilities()` matched
Walmart's *"Total liabilities, redeemable noncontrolling interest, and equity"*
— which equals total **assets** by the balance-sheet identity. `get_revenue()`
returned Pfizer's product revenues (50,914) against total revenues of 58,496. So
nothing is trusted on the getter's word: every figure is checked against the
statement line it should have come from, and anything that doesn't reconcile is
stored as **missing, never as a number**. A gap is visible and gets fixed; a
plausible wrong number is neither.

**Why does the retrieval tool only handle one filing at a time?** It forces
decomposition to be explicit. A three-year trend becomes three calls and the
model compares the figures itself, rather than a single vague search that
returns a mix of years. It also means every retrieval is scoped, which is what
makes citations precise.

**Why validate the LLM's arguments instead of trusting them?** A model will
confidently pass `ticker="Apple"`, `ticker="APPL"`, or `year=2025`. A hard
failure ends the run, so the validator fuzzy-matches, clamps, or drops the
filter — and **records every adjustment**, which is surfaced back to the model.
`top_k` is deliberately *not* exposed: the model has no basis for choosing a
value, and every extra tool parameter is one more thing it can emit malformed.

**Why a relevance floor?** A human reading a 0.001-relevance chunk ignores it; a
model cites it. Chunks below 0.3 are dropped before the model sees them, and the
tool distinguishes three states — nothing matched the filters, matched but all
below floor, or results. The middle one carries an explicit instruction *not* to
answer from them, because the failure being guarded against is a model filling a
silence with training data.

**Why key the fiscal year off `period_of_report`?** Apple's FY2024 10-K covers a
period ending September 2024 but was filed in November. Keying off the period is
what makes "year" comparable across filers with different calendars. A
second-order fix sits on top: Home Depot names a fiscal year after its *starting*
calendar year while Walmart names it after the ending one.

---

## Guardrails

The system prompt *asks* the model to stay grounded. The guardrails *verify* it.
All four checks are deterministic — no second LLM call — because a model that
invented `$391,035M` will confirm it when asked to check its own work.
Self-critique fails precisely where factual errors live.

**Before the agent runs**

| Check | What it does |
|---|---|
| `scope` | Refuses years outside FY2022–24; warns when no in-corpus company is named |
| `advice` | Refuses "should I buy", price targets, predictions — nothing in a 10-K answers them |

**After the answer comes back**

| Check | What it does |
|---|---|
| `numeric_grounding` | Every figure must appear in a retrieved chunk, come from `get_financials`, or be derivable from numbers that do |
| `citations` | Every cited chunk must have actually been retrieved on this turn |

```
$ python -m agent.run_agent "How did Apple's revenue change from FY2022 to FY2024?"

  GUARDRAILS
  figures: 2 exact, 0 rescaled, 2 derived, 0 unsupported (2 skipped as years/ordinals)
  clean
```

```
$ python -m agent.run_agent "Should I buy Apple stock?"

  REFUSED
  This system reports what SEC 10-K filings say; it does not give investment
  advice or price predictions.
```

Run the checks: `python -m guardrails.test_guardrails` — 29 cases, no pytest
dependency.

### Why these and not a jailbreak classifier?

The corpus is public, audited SEC filings. The prompt-injection surface is
theoretical, so a jailbreak filter would be copied from a system with a
different threat model. The failure that actually happens here is a confidently
wrong number, so that is what gets checked — with arithmetic.

### Three cases that decide whether this check is useful

**A derived figure is not a hallucination.** "Revenue fell $3,293 million" is
`394,328 − 391,035`, both retrieved. Reporting that as invented is the single
easiest way to make the check worthless, so derivation is attempted — differences,
sums, ratios and percentage changes — before anything is called invented.

**Scale words have to match bare table figures.** A 10-K states scale once, in a
heading ("in millions"), then prints bare numbers. A chunk carries `394,328`
while the answer writes `$394,328 million` or `$394.3 billion`. Each figure is
therefore carried as both readings and matched on either — without this, correct
arithmetic gets flagged.

**Percentages are never matched on significant digits.** Currency rescales
legitimately; a percentage does not. `8%` is not evidence for `80%`, and
accepting it would pass a tenfold error — the exact failure this exists to
catch. Percentages must be exact or derived.

### What it does not do

It cannot catch a wrong claim made without a number ("Apple's risks are
primarily regulatory" when the filing says supply chain), and the derivation
search gets more permissive as the pool of retrieved figures grows — with enough
numbers in context, a coincidental arithmetic match becomes likelier. It errs
toward allowing, which is the right bias for a check that blocks answers, but it
is a ceiling on what this can prove.

The same rules run over the eval set **are** the answer-faithfulness eval:
`eval/faithfulness_eval.py`. One answer at request time, 65 in the harness.

---

## Failure Modes

| # | Symptom | Root cause | Status |
|---|---|---|---|
| FM-1 | Table chunks lose to mediocre prose | Tables carry no BM25 signal; a dense-only chunk scores `1/61` and loses to any dual-list chunk | Fixed — v4 cross-encoder |
| FM-2 | Item 1A crowded out of results | JNJ has 145 Item 8 chunks to 19 Item 1A; a flat top-20 is 20 Item 8 and zero Item 1A | Fixed — section quota (unmeasured) |
| FM-3 | Sparse sections noisy or unreachable | "None." sections indistinguishable from real ones | Fixed — `is_sparse` + `must_not` |
| FM-4 | XOM MD&A questions return Item 16 | XOM answers Item 7/8 with "Reference is made to…" and appends the Financial Section after Item 16 | Fixed in parser — **needs re-ingest** |
| FM-5 | WMT FY2024 content indexed twice | Item 4 ran on for 87,852 chars, swallowing Part II; boundary detection missed `ITEM 5.MARKET` | Fixed in parser — **needs re-ingest** |
| FM-6 | Confidently wrong financial figures | `companyfacts` dimensional members; then getters picking neighbouring lines | Fixed — reconciliation gate |
| FM-7 | Stale chunks survive re-ingest | Point ID includes section, so moved content writes new points — and there is no delete path | **Open** |
| FM-8 | `top1_score` incomparable across versions | `rerank()` overwrites `result.score`; the eval records it blind | **Open** |
| FM-9 | Stored eval results unreproducible | 15 questions added after the runs; all files say `n=50` | **Open** |
| FM-10 | Agent pins sections the eval says are harmful | Two decisions made at different times; the eval never exercises the agent path | **Open** |
| FM-11 | Groq emits malformed tool calls | Generation fluke — `<function=…>` as text, rejected 400 | Mitigated — 3 retries + fallback model |
| FM-12 | Long chunks truncated at rerank | A 2,200-char chunk exceeds 512 tokens; the tail is never scored | **Open**, documented |

---

## Evaluation

65 questions across all 10 companies, hand-labelled with expected answer
keywords and tagged by type and difficulty.

**`keyword_hit@k`** — of a question's expected keywords, what fraction appear
anywhere in the top-k chunks. Deliberately generous: it asks *did retrieval put
the answer in front of the model*, not *did the model answer correctly*, which
is the right question for a layer with no LLM in it.

**`ticker_hit` / `year_hit` / `section_hit`** separate "wrong document" from
"right document, wrong part of it". `section_hit` is the discriminating one for
finding failures.

`top1_score` is recorded but **excluded from the results table on purpose**: it
is cosine similarity in v1, an RRF score in v2/v3, and a cross-encoder
probability in v4, because `rerank()` overwrites `result.score`. Read across
versions it looks like a 29× improvement; it's a change of units.

### Answer faithfulness

`eval/faithfulness_eval.py` runs the agent over the ground-truth questions and
applies the same rules the guardrails apply at request time — same checks, 65
answers instead of one. It is the only part of the eval that exercises the agent
path.

```bash
# generate answers and judge them
python eval/faithfulness_eval.py --limit 10 \
       --runs eval/results/runs.json --output eval/results/faithfulness.json

# re-judge the SAME answers after changing guardrails/rules.py — no agent calls
python eval/faithfulness_eval.py --judge-only --runs eval/results/runs.json
```

The two phases are separable on purpose. Generating answers is slow, costs Groq
calls and is not perfectly repeatable; judging is free and deterministic.
Splitting them means a change to the rules is re-scored against answers that
already exist, and a bug in the judge cannot consume the expensive half of the
work.

Reported per answer and in aggregate:

| Metric | Meaning |
|---|---|
| `faithful_rate` | answers with no blocking finding |
| `invented_figure_rate` | answers stating a figure that is in no chunk and derivable from none |
| `bad_citation_rate` | answers citing a source that was never retrieved |
| `wrong_chunk_index_rate` | right filing and section, wrong chunk index |
| `no_citation_rate` | chunks were retrieved and the answer cites none |
| `figures.*` | every figure split into exact / rescaled / derived / invented |

A question that errors is recorded with its error rather than dropped — a
harness that silently skips failures reports a better pass rate than the system
earns.

**Not yet run against the real corpus.** It needs Qdrant up and a Groq key; the
judging half is covered by the guardrail tests.

### What it doesn't measure

- **Claims made without a number.** "Apple's risks are primarily regulatory"
  when the filing says supply chain is unfalsifiable by these rules.
- **Retrieval quality, from the agent's side.** `eval/run_eval.py` still calls
  `search()` directly, so the v1–v4 numbers describe the pre-agent system. This
  is why FM-10 went unnoticed.
- **n=65 is small.** 95% CIs are roughly ±10pp — wider than several of the
  deltas above. v1→v3 is probably real; v3→v4 probably isn't.
- **Ground truth is unverified.** At least one confirmed error:
  `XOM-2023-7-002` expects `26,279`, which appears nowhere in the filing — XOM
  writes "26.3 billion".
- **Two result files measure nothing.** `v1_*.prefix.json` claim to test the bge
  query prefix; the two runs are retrieval-identical. The differences come from
  the ground truth being edited between runs.

---

## Current State

| Component | State |
|---|---|
| Ingest → Qdrant | Working |
| Retrieval v1–v4 | Working, measured |
| Section quota | Implemented, not yet measured end to end |
| XBRL financials | Working, measured on 240 extractions |
| ReAct agent | Working, never measured by the *retrieval* eval |
| Guardrails (4 checks) | Working, 31 tests |
| Answer-faithfulness eval | Built; not yet run against the real corpus |

**Two things must happen before the numbers above can be trusted again:**

1. **Re-ingest needs a delete step first (FM-7).** The parser fix moves content
   between sections. Point IDs are `uuid5(filing_id|section|chunk_index)`, so
   moved content writes *new* points and the stale ones survive — `ingest.py`
   has no delete path. Drop the collection or delete by `filing_id` first, or
   the corrected corpus will be worse than the broken one.
2. **The stored eval results are stale.** Every file in `eval/results/` reports
   `num_questions=50`; the dataset is now 65.

**Next:** delete path → re-ingest → re-run all configs at n=65 → run
`faithfulness_eval.py` over the full set and record a baseline.

---

## Development Log

Reconstructed from file creation times; commit dates match.

**2026-07-29 · Ingest.** EDGAR fetch, section extraction, token-aware chunking,
Qdrant upsert. Three parse bugs found and fixed here: page-footer bleed
(`Apple Inc. | 2022 Form 10-K | 19` leaking into sections that straddled a page
break), the signature block appending to the last Item, and a heading stripper
that ate whole section bodies because EDGAR sometimes runs a section title
straight into the first sentence.

**2026-07-29 · Retrieval v1.** Dense cosine, Phoenix tracing, CLI. Baseline
**78.0%**. The anomaly that set the agenda: *easy* questions scored **below**
medium ones. Easy questions are exact-numeral lookups — dense retrieval's
weakest case.

**2026-07-30 · Retrieval v2.** BM25 + RRF, aimed squarely at that weakness.
**78.0% → 84.5%**, zero-hit rate to 0.00.

**2026-07-31 · Retrieval v3.** Structured query parsing. **84.5% → 86.8%**, and
perfect-recall 50% → 62% — the largest gain in the project, from *not* using a
model.

**2026-07-31 · Retrieval v4.** Cross-encoder reranking. **86.8% → 87.5%**,
section hit 0.84 → 0.88, latency 116 ms → 7,153 ms. A shallower pool (10 instead
of 20) was tried and **reverted**: it was justified by two questions rescued from
RRF ranks ≤10, but RRF buries Item 1A below rank 10 whenever Item 7/8 crowd the
pool, so risk-factor questions lost their Item 1A evidence entirely. The reason
is recorded in `config.py` so it isn't retried.

**2026-07-31 · Agent.** ReAct loop, argument validation, relevance floor,
LangSmith tracing. The agent is a front end over a stack that was already
finished and measured — `search()`'s signature is unchanged and `retrieve.py`
still works. Phoenix is disabled on the agent path: its export is synchronous
and deadlocks against LangGraph's worker threads. `TORCH_DEVICE` is forced to
`cpu` for the same reason — MPS plus LangGraph's threads/forks segfaults the
interpreter.

**2026-08-21 · Eval expanded 50 → 65.** The original set had a structural bias:
**33 of 50 questions targeted Item 7**, which is what the system is already best
at, so aggregate keyword hit was saturated and blind to conceptual sections. 15
section-targeted questions were added, including two **intentional canaries** on
Item 1B — a section whose entire content is "None.", flagged `is_sparse` and
excluded from retrieval. They must score 0% `section_hit` while that policy is
on; they exist to keep it visible.

**2026-09-13 · Parser fixes, section quota, XBRL financials.** The parser bug
was found by reading per-question eval output — every chunk returned for XOM's
MD&A questions was labelled `Item 16`. Tracing didn't find it; the eval did, and
had been flagging it for weeks as low section accuracy that was read as a
ranking problem. Measured over all 30 filings: WMT 2024 Item 4 went 51 chunks →
1, XOM 2022/23 Item 7 went from a 1-chunk stub to ~134K/139K chars, and the
other 27 filings were unchanged.

---

## A Note on Scope

This is a research and measurement project, not a product. It's deliberately
small — 10 companies, 3 years, one corpus — because the goal was to make each
retrieval decision measurable rather than to scale. Where a decision was made on
intuition rather than measurement, this README says so.

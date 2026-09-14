"""Central configuration for the SEC 10-K ingestion pipeline."""

import os

# --- EDGAR -------------------------------------------------------------------
# The SEC requires a descriptive User-Agent with a contact address on every
# request. Override with SEC_IDENTITY if you want your own address in the logs.
SEC_IDENTITY = os.environ.get("SEC_IDENTITY", "")

COMPANIES = ["AAPL", "MSFT", "JPM", "JNJ", "WMT", "XOM", "GOOGL", "BAC", "PFE", "HD"]

# Fiscal years, keyed off the filing's period_of_report (not the filing date).
# AAPL's FY2024 10-K covers period 2024-09-28 but was filed in Nov 2024; keying
# off the period is what makes "year" comparable across companies with different
# fiscal calendars.
YEARS = [2022, 2023, 2024]

# --- Embeddings --------------------------------------------------------------
# bge-small-en-v1.5 is a drop-in swap for all-MiniLM-L6-v2: also 384-dim, but
# with a 512-token max sequence length so 512-token chunks are not truncated.
# Swap freely -- EMBEDDING_DIM is verified against the loaded model at runtime.
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "384"))
EMBEDDING_BATCH_SIZE = 64

# None = let torch pick (MPS on Apple silicon). The agent forces "cpu": MPS plus
# LangGraph's threads/forks segfaults the interpreter, and these models are small
# enough that CPU inference costs little.
TORCH_DEVICE = os.environ.get("TORCH_DEVICE") or None

# bge models are trained asymmetrically: passages are embedded bare, queries get
# a short instruction prefix. Ingest stores passages bare, so this applies to
# queries only. Set to "" to disable and embed queries bare as well.
QUERY_INSTRUCTION = os.environ.get(
    "QUERY_INSTRUCTION",
    "Represent this sentence for searching relevant passages: ",
)

# --- Chunking ----------------------------------------------------------------
# Measured in tokens of the embedding model's own tokenizer, so a chunk is
# exactly what the model sees.
CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 64

# A section this short is boilerplate ("None.", "Not applicable.") rather than
# substantive disclosure. Still ingested, but flagged is_sparse for the
# negative-retrieval evals.
SPARSE_CHAR_THRESHOLD = 200

# --- Qdrant ------------------------------------------------------------------
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "sec_10k"
# The BM25 build scrolls the whole collection; payloads are stored on disk, so a
# cold page cache can push a single scroll well past the client default.
QDRANT_TIMEOUT = int(os.environ.get("QDRANT_TIMEOUT", "120"))
UPSERT_BATCH_SIZE = 256

# --- Retrieval ---------------------------------------------------------------
DEFAULT_TOP_K = 5

# "v1" = dense only. "v2" = dense + BM25 fused with Reciprocal Rank Fusion.
DEFAULT_RETRIEVAL = os.environ.get("DEFAULT_RETRIEVAL", "v1")

# How deep each retriever goes before fusion. Deeper candidate pools give RRF
# more to work with; k=5 final results out of 50+50 candidates.
DENSE_CANDIDATES = 50
BM25_CANDIDATES = 50

# RRF damping constant. 60 is the value from the original Cormack et al. paper
# and is what most implementations use; larger flattens the rank weighting.
RRF_K = 60

# Weight applied to BM25's contribution inside RRF.
# Default 1.0 = BM25 and dense carry equal weight (standard RRF).
# Values < 1.0 demote BM25 from a hard co-occurrence gate to a soft boost:
# at w=0.5 the worst dual-list score (0.5/110=0.00455) falls below the best
# dense-only score (1/61=0.01639), breaking the intersection gate.
# Breaking point: w < 0.80. Override via --bm25-weight in run_eval.py or
# by setting BM25_WEIGHT directly before calling search_hybrid().
BM25_WEIGHT = float(os.environ.get("BM25_WEIGHT", "1.0"))

# --- v4 cross-encoder reranking ----------------------------------------------
# Same family as the bi-encoder, so query/passage conventions match.
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-base")
# How many v3 candidates to re-score before cutting to top_k. Deeper gives the
# reranker more chances to rescue a chunk that RRF buried.
RERANK_CANDIDATES = 20
RERANK_BATCH_SIZE = 32

# --- section quota (candidate-pool budgeting) --------------------------------
# The largest share of the rerank pool any single 10-K section may occupy. The
# imbalance this corrects is structural: JNJ FY2023 has 145 Item 8 chunks to 19
# Item 1A chunks, so a flat top-N fills with Item 8 on volume before relevance
# is considered. 0.5 lets Item 8 hold at most half the pool. Set to 1.0 to
# disable and restore flat top-N behaviour.
SECTION_QUOTA_SHARE = float(os.environ.get("SECTION_QUOTA_SHARE", "0.5"))
# The quota reallocates slots rather than discarding them, so the underlying
# draw is widened by this factor to give it something to reallocate *from*.
# Costs a deeper Qdrant/BM25 read, not a deeper rerank -- the pool is cut back
# to RERANK_CANDIDATES before the cross-encoder sees it.
QUOTA_POOL_FACTOR = int(os.environ.get("QUOTA_POOL_FACTOR", "3"))

# Cross-encoder scores below this are noise, not evidence. A chunk at 0.001 did
# not contribute to any answer, so it must not reach the LLM or the source list.
RELEVANCE_FLOOR = float(os.environ.get("RELEVANCE_FLOOR", "0.3"))

# --- Agent -------------------------------------------------------------------
# Must match the eval's RERANK_CANDIDATES. A shallower pool was tried (10) as a
# latency optimisation, justified by the two eval questions that were rescued
# from RRF ranks <= 10 -- but that does not generalise: RRF buries Item 1A chunks
# below rank 10 whenever the filing's huge Item 7/Item 8 sections crowd the pool,
# so "risk factors" questions lost their Item 1A evidence entirely. Passed
# explicitly into search() -- never by mutating RERANK_CANDIDATES, which would
# silently change eval runs in any process that imports the agent.
AGENT_RERANK_CANDIDATES = 20
# Not exposed to the LLM: it has no basis for choosing a value, and every extra
# tool parameter is one more thing a model can emit malformed.
AGENT_TOP_K = 5

# --- v3 structured query parsing ---------------------------------------------
COMPANY_ALIASES = {
    "AAPL": ["apple inc", "apple"],
    "MSFT": ["microsoft corporation", "microsoft"],
    "JPM": ["jpmorgan chase", "jpmorgan", "jp morgan", "j.p. morgan", "chase"],
    "JNJ": ["johnson & johnson", "johnson and johnson", "j&j", "johnson"],
    "WMT": ["walmart inc", "walmart", "wal-mart"],
    "XOM": ["exxonmobil", "exxon mobil", "exxon"],
    "GOOGL": ["alphabet inc", "alphabet", "google"],
    "BAC": ["bank of america", "bofa", "b of a"],
    "PFE": ["pfizer inc", "pfizer"],
    "HD": ["home depot", "the home depot"],
}

# Corpus years are keyed off period_of_report, but filers label their fiscal
# years inconsistently. Home Depot names a fiscal year after its STARTING
# calendar year -- HD "fiscal 2023" ended 2024-01-28 and is tagged year=2024
# here. Walmart names its fiscal year after the ENDING calendar year, so WMT
# "fiscal 2023" ended 2023-01-31 and needs no shift. Everyone else is aligned.
# offset = corpus_year_tag - fiscal_year_label_in_the_query
FISCAL_YEAR_OFFSET = {"HD": 1}

# Filtering on a parsed section is OFF by default: measured on this corpus, most
# answer-bearing chunks do NOT live in the section the question nominally targets
# (revenue figures sit in Item 8 tables even for Item 7 questions), so a hard
# section filter removes more right answers than wrong ones. parse_query still
# returns the section for callers that want it.
V3_SECTION_FILTER = False

# Search using the entity-stripped query text rather than the raw question.
# OFF: the ticker/year filters already constrain the search to one filing, so
# stripping those tokens from the text solves nothing the filter has not already
# solved -- while throwing away signal that still matters INSIDE the filtered
# set (a 10-K discusses three years of columns, so "fiscal 2023 vs 2022" tells
# the retriever which rows matter). Kept as a flag because parse_query still
# returns clean_query for callers that want it.
V3_USE_CLEAN_QUERY = False

# --- Observability (Phoenix) -------------------------------------------------
# Phoenix export is synchronous (batch=False). That is fine for a single-threaded
# CLI, but inside LangGraph the tool runs on a worker thread while the main
# thread waits on it -- a blocked export deadlocks both. The agent turns this off
# and relies on LangSmith instead.
PHOENIX_ENABLED = os.environ.get("PHOENIX_ENABLED", "1").lower() not in ("0", "false", "no")
PHOENIX_PORT = int(os.environ.get("PHOENIX_PORT", "6006"))
PHOENIX_PROJECT = os.environ.get("PHOENIX_PROJECT", "sec-rag-retrieval")
PHOENIX_ENDPOINT = os.environ.get(
    "PHOENIX_ENDPOINT", f"http://localhost:{PHOENIX_PORT}/v1/traces"
)

# --- Sections ----------------------------------------------------------------
# (item, label, part, optional). `optional` means "absent from older filings by
# design" -- skip silently rather than warning.
SECTIONS = [
    ("Item 1", "Business", "Part I", False),
    ("Item 1A", "Risk Factors", "Part I", False),
    ("Item 1B", "Unresolved Staff Comments", "Part I", False),
    ("Item 1C", "Cybersecurity", "Part I", True),  # 2023+ filings only
    ("Item 2", "Properties", "Part I", False),
    ("Item 3", "Legal Proceedings", "Part I", False),
    ("Item 4", "Mine Safety Disclosures", "Part I", False),
    ("Item 5", "Market for Registrant's Equity", "Part II", False),
    ("Item 6", "Reserved", "Part II", False),
    ("Item 7", "MD&A", "Part II", False),
    ("Item 7A", "Quantitative Disclosures About Market Risk", "Part II", False),
    ("Item 8", "Financial Statements", "Part II", False),
    ("Item 9", "Changes in and Disagreements with Accountants", "Part II", False),
    ("Item 9A", "Controls and Procedures", "Part II", False),
    ("Item 9B", "Other Information", "Part II", False),
    ("Item 9C", "Disclosure Regarding Foreign Jurisdictions", "Part II", True),
    ("Item 10", "Directors, Executive Officers and Corporate Governance", "Part III", False),
    ("Item 11", "Executive Compensation", "Part III", False),
    ("Item 12", "Security Ownership of Certain Beneficial Owners", "Part III", False),
    ("Item 13", "Certain Relationships and Related Transactions", "Part III", False),
    ("Item 14", "Principal Accountant Fees and Services", "Part III", False),
    ("Item 15", "Exhibits and Financial Statement Schedules", "Part IV", False),
    ("Item 16", "Form 10-K Summary", "Part IV", True),
]

SECTION_LABELS = {item: label for item, label, _part, _opt in SECTIONS}
SECTION_PARTS = {item: part for item, _label, part, _opt in SECTIONS}
OPTIONAL_SECTIONS = {item for item, _l, _p, opt in SECTIONS if opt}

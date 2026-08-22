# eval/dataset.py
# ─────────────────────────────────────────────────────────────────────────────
# RAG Eval Ground-Truth Dataset — SEC 10-K Corpus
#
# Coverage : AAPL, MSFT, JPM, JNJ, WMT, XOM, GOOGL, BAC, PFE, HD
# Fisc. yrs: 2022 – 2024
# Questions: 50 total, 5 per ticker
#
# VERIFICATION NOTE
# answer_keywords are drawn from the author's knowledge of public SEC filings.
# Before deploying in production, spot-check every numerical keyword against
# the actual EDGAR HTML to confirm it appears verbatim in the filing
# (all dollar amounts are in millions unless the question states otherwise).
# Lines marked  # ⚠ verify  are the figures most likely to need a quick check.
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass
from typing import List


@dataclass
class EvalQuestion:
    id: str
    question: str
    ticker: str
    year: int
    section: str
    answer_keywords: List[str]
    question_type: str   # "factual" | "numerical" | "comparative" | "trend"
    difficulty: str      # "easy" | "medium" | "hard"
    notes: str


QUESTIONS = [

    # ══════════════════════════════════════════════════════════════════════════
    # AAPL  –  FY2022 (ended Sep 24 2022) · FY2023 (ended Sep 30 2023)
    # ══════════════════════════════════════════════════════════════════════════

    EvalQuestion(
        id="AAPL-2023-7-001",
        question="What were Apple's total net sales for fiscal year 2023?",
        ticker="AAPL",
        year=2023,
        section="Item 7",
        answer_keywords=["383,285", "net sales", "fiscal 2023", "total"],
        question_type="numerical",
        difficulty="easy",
        notes="Top-line revenue lookup; tests basic retrieval of the MD&A consolidated results table.",
    ),
    EvalQuestion(
        id="AAPL-2023-7-002",
        question="How did Apple's Services segment revenue change between fiscal year 2022 and fiscal year 2023?",
        ticker="AAPL",
        year=2023,
        section="Item 7",
        answer_keywords=["services", "85,200", "78,129", "increased"],
        question_type="comparative",
        difficulty="medium",
        notes="Two-year Services comparison; the system must retrieve both figures from the product/service revenue table and read the direction.",
    ),
    EvalQuestion(
        id="AAPL-2022-7-001",
        question="What was Apple's iPhone net sales for fiscal year 2022?",
        ticker="AAPL",
        year=2022,
        section="Item 7",
        answer_keywords=["205,489", "iphone", "net sales", "2022"],
        question_type="numerical",
        difficulty="easy",
        notes="Product-level revenue row; iPad/Mac/iPhone rows are adjacent in the table, testing exact row-level precision.",
    ),
    EvalQuestion(
        id="AAPL-2023-1A-001",
        question="What geographic manufacturing concentration risk does Apple highlight in its fiscal 2023 risk factors?",
        ticker="AAPL",
        year=2023,
        section="Item 1A",
        answer_keywords=["china", "manufacturing", "concentration"],
        question_type="factual",
        difficulty="medium",
        notes="Named-country risk buried within a long 1A section; tests whether retrieval routes to the correct risk paragraph.",
    ),
    EvalQuestion(
        id="AAPL-2023-7-003",
        question="What was Apple's overall gross margin percentage for fiscal year 2023, and how did it compare to fiscal year 2022?",
        ticker="AAPL",
        year=2023,
        section="Item 7",
        answer_keywords=["44.1", "43.3", "gross margin", "percentage"],
        question_type="trend",
        difficulty="hard",
        notes="Hard: answer is expressed as two percentages, not dollar figures; both years must be present in the retrieved chunk.",
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # MSFT  –  FY2023 (ended Jun 30 2023) · FY2024 (ended Jun 30 2024)
    # ══════════════════════════════════════════════════════════════════════════

    EvalQuestion(
        id="MSFT-2023-7-001",
        question="What was Microsoft's Intelligent Cloud segment revenue for fiscal year 2023?",
        ticker="MSFT",
        year=2023,
        section="Item 7",
        answer_keywords=["87,907", "intelligent cloud", "revenue", "fiscal 2023"],
        question_type="numerical",
        difficulty="easy",
        notes="Segment revenue lookup; tests whether retrieval surfaces the correct row from the three-segment breakdown table.",
    ),
    EvalQuestion(
        id="MSFT-2023-7-002",
        question="How did Microsoft's total revenue change from fiscal year 2022 to fiscal year 2023?",
        ticker="MSFT",
        year=2023,
        section="Item 7",
        answer_keywords=["211,915", "198,270", "revenue", "increased"],
        question_type="comparative",
        difficulty="medium",
        notes="Year-over-year top-line comparison; both figures should appear in the MD&A overview paragraph.",
    ),
    EvalQuestion(
        id="MSFT-2024-1A-001",
        question="What AI-related competitive risk does Microsoft identify in its fiscal year 2024 risk factors?",
        ticker="MSFT",
        year=2024,
        section="Item 1A",
        answer_keywords=["artificial intelligence", "generative AI", "competition", "cloud"],
        question_type="factual",
        difficulty="medium",
        notes="Verbatim from FY2024 Item 1A 'Business model competition' section; 'Copilot' does not appear in this risk section — 'generative AI' is the actual term used.",
    ),
    EvalQuestion(
        id="MSFT-2024-7-001",
        question="What was Microsoft's total revenue for fiscal year 2024?",
        ticker="MSFT",
        year=2024,
        section="Item 7",
        answer_keywords=["245,122", "total revenue"],
        question_type="numerical",
        difficulty="easy",
        notes="FY2024 top-line; verifies the system fetches the 2024 filing rather than the more commonly retrieved 2023 document.",
    ),
    EvalQuestion(
        id="MSFT-2023-7-003",
        question="Which of Microsoft's three business segments generated the lowest revenue in fiscal year 2023?",
        ticker="MSFT",
        year=2023,
        section="Item 7",
        answer_keywords=["more personal computing", "54,734", "segment", "revenue"],
        question_type="numerical",
        difficulty="hard",
        notes="Hard: the system must compare all three segment figures to identify the minimum; a chunk containing only one segment will fail.",
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # JPM  –  FY2022 · FY2023
    # ══════════════════════════════════════════════════════════════════════════

    EvalQuestion(
        id="JPM-2022-7-001",
        question="How did JPMorgan Chase's provision for credit losses change from 2021 to 2022, and what macro factors drove that change?",
        ticker="JPM",
        year=2022,
        section="Item 7",
        answer_keywords=["provision for credit losses", "macroeconomic outlook", "mild recession", "net addition"],
        question_type="comparative",
        difficulty="medium",
        notes="Verbatim from 2022 10-K Item 7; 'reserve build' and 'economic uncertainty' do not appear — actual language is 'macroeconomic outlook', 'mild recession', 'net addition to the allowance'.",
    ),
    EvalQuestion(
        id="JPM-2022-7-002",
        question="What was JPMorgan Chase's net income for full-year 2022?",
        ticker="JPM",
        year=2022,
        section="Item 7",
        answer_keywords=["net income", "37,676", "2022", "common stockholders"],
        question_type="numerical",
        difficulty="medium",
        notes="Bottom-line profitability; tests precise figure retrieval from a densely populated bank income statement.",
    ),
    EvalQuestion(
        id="JPM-2023-7A-001",
        question="What risk-quantification methodology does JPMorgan Chase use in its 2023 Item 7A market risk disclosures?",
        ticker="JPM",
        year=2023,
        section="Item 7A",
        answer_keywords=["value at risk", "var", "95%", "market risk"],
        question_type="factual",
        difficulty="medium",
        notes="Tests retrieval from Item 7A, a less-commonly indexed section; verifies section-boundary awareness in chunking strategies.",
    ),
    EvalQuestion(
        id="JPM-2023-7A-002",
        question="How does JPMorgan describe the interest rate sensitivity of its net interest income in the 2023 quantitative market risk disclosures?",
        ticker="JPM",
        year=2023,
        section="Item 7A",
        answer_keywords=["net interest income", "basis points", "sensitivity", "parallel shift"],
        question_type="factual",
        difficulty="hard",
        notes="Hard: rate-sensitivity tables in 7A are numerically dense; the system must surface the NII-sensitivity table rather than generic rate-risk language.",
    ),
    EvalQuestion(
        id="JPM-2022-7-003",
        question="What was JPMorgan Chase's total net revenue for 2022, and how did it compare to 2021?",
        ticker="JPM",
        year=2022,
        section="Item 7",
        answer_keywords=["net revenue", "128,695", "2022", "increased"],
        question_type="comparative",
        difficulty="medium",
        notes="Managed-basis net revenue is distinct from GAAP revenue for banks; tests whether the system retrieves the correct metric.",
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # JNJ  –  FY2023 · FY2024
    # ══════════════════════════════════════════════════════════════════════════

    EvalQuestion(
        id="JNJ-2023-1-001",
        question="What two business segments did Johnson & Johnson operate following the completion of the Kenvue separation in 2023?",
        ticker="JNJ",
        year=2023,
        section="Item 1",
        answer_keywords=["innovative medicine", "medtech", "kenvue", "separation"],
        question_type="factual",
        difficulty="medium",
        notes="Structural change question; the Kenvue Consumer spin-off fundamentally altered JNJ's reportable segments.",
    ),
    EvalQuestion(
        id="JNJ-2023-1-002",
        question="Approximately how many employees did Johnson & Johnson have worldwide at the end of fiscal year 2023?",
        ticker="JNJ",
        year=2023,
        section="Item 1",
        answer_keywords=["131,900", "employees", "worldwide"],
        question_type="numerical",
        difficulty="easy",
        notes="Verbatim from FY2023 10-K Item 1; '131,900 employees worldwide' appears in the General and Geographic Areas sections. 'Human capital' does not appear in this filing.",
    ),
    EvalQuestion(
        id="JNJ-2023-1-003",
        question="What therapeutic areas does Johnson & Johnson identify as primary research focus areas within its Innovative Medicine segment in 2023?",
        ticker="JNJ",
        year=2023,
        section="Item 1",
        answer_keywords=["oncology", "immunology", "neuroscience", "innovative medicine"],
        question_type="factual",
        difficulty="hard",
        notes="Hard: requires identifying multiple named therapeutic areas spread across a lengthy Item 1 pipeline description; no single paragraph contains all of them.",
    ),
    EvalQuestion(
        id="JNJ-2024-7-001",
        question="What was Johnson & Johnson's total worldwide sales for full-year 2024?",
        ticker="JNJ",
        year=2024,
        section="Item 7",
        answer_keywords=["88,821", "88.8 billion", "worldwide sales", "4.3%"],
        question_type="numerical",
        difficulty="easy",
        notes="Verbatim from FY2024 10-K Item 7: 'worldwide sales increased 4.3% to $88.8 billion'. Post-Kenvue figure; tests that the system retrieves the 2024 filing rather than the more widely indexed 2023 document.",
    ),
    EvalQuestion(
        id="JNJ-2024-7-002",
        question="How did Johnson & Johnson's MedTech segment sales compare between fiscal year 2023 and fiscal year 2024?",
        ticker="JNJ",
        year=2024,
        section="Item 7",
        answer_keywords=["31,857", "30,400", "4.8%", "medtech"],
        question_type="comparative",
        difficulty="medium",
        notes="Verbatim from FY2024 10-K Item 7 segment table; MedTech 2024=$31,857M vs 2023=$30,400M, +4.8%. Tests row-level precision — adjacent to Innovative Medicine row in the same table.",
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # WMT  –  FY2022 (ended Jan 28 2022) · FY2023 (ended Jan 27 2023)
    # ══════════════════════════════════════════════════════════════════════════

    EvalQuestion(
        id="WMT-2022-1-001",
        question="How many Walmart U.S. stores were in operation at the end of fiscal year 2022?",
        ticker="WMT",
        year=2022,
        section="Item 1",
        answer_keywords=["4,742", "walmart u.s.", "stores", "fiscal 2022"],
        question_type="numerical",
        difficulty="easy",
        notes="Store count from the Item 1 operations table; tests retrieval of a simple numerical property from a properties-style list.",
    ),
    EvalQuestion(
        id="WMT-2022-1-002",
        question="In how many countries did Walmart International operate at the end of fiscal year 2022?",
        ticker="WMT",
        year=2022,
        section="Item 1",
        answer_keywords=["countries", "walmart international", "19"],
        question_type="numerical",
        difficulty="easy",
        notes="Geographic footprint count from Item 1 International segment description; tests retrieval of a non-U.S. operational metric.",
    ),
    EvalQuestion(
        id="WMT-2023-7-001",
        question="What were Walmart's total revenues for fiscal year 2023?",
        ticker="WMT",
        year=2023,
        section="Item 7",
        answer_keywords=["611,289", "total revenues", "fiscal 2023"],
        question_type="numerical",
        difficulty="easy",
        notes="Headline revenues figure ($611,289M = net sales $605,881M + membership/other income); Walmart's January fiscal year-end is a common year-tagging trap for RAG systems.",
    ),
    EvalQuestion(
        id="WMT-2023-7-002",
        question="What was Walmart U.S. comparable store sales growth (including fuel) in fiscal year 2023?",
        ticker="WMT",
        year=2023,
        section="Item 7",
        answer_keywords=["comparable sales", "7.0%", "walmart u.s.", "fiscal 2023"],
        question_type="comparative",
        difficulty="medium",
        notes="Comp sales KPI is a distinct retail metric from total sales; with-fuel figure (7.0%) appears verbatim in the comparable-sales table. Tests whether the system retrieves the specific comparable-store discussion.",
    ),
    EvalQuestion(
        id="WMT-2023-7-003",
        question="How did Walmart's consolidated operating income change between fiscal year 2022 and fiscal year 2023, and what drove the change?",
        ticker="WMT",
        year=2023,
        section="Item 7",
        answer_keywords=["operating income", "20,428", "25,942", "fiscal 2023", "decreased"],
        question_type="trend",
        difficulty="hard",
        notes="Hard: operating income declined ($25,942M → $20,428M) in FY2023 despite strong revenue growth — a non-obvious result; the system must retrieve the profitability discussion, not just the revenue table.",
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # XOM  –  FY2023 · FY2024
    # (No Item 6 in any XOM year — no questions target that section.)
    # ══════════════════════════════════════════════════════════════════════════

    EvalQuestion(
        id="XOM-2023-7-001",
        question="What were ExxonMobil's total earnings in 2023?",
        ticker="XOM",
        year=2023,
        section="Item 7",
        answer_keywords=["36,010", "earnings", "2023"],
        question_type="numerical",
        difficulty="easy",
        notes="Headline earnings figure from MD&A; tests basic financial retrieval from a large integrated oil company filing.",
    ),
    EvalQuestion(
        id="XOM-2023-7-002",
        question="What were ExxonMobil's capital and exploration expenditures in 2023?",
        ticker="XOM",
        year=2023,
        section="Item 7",
        answer_keywords=["capital", "exploration", "expenditures", "26,279", "2023"],
        question_type="numerical",
        difficulty="medium",
        notes="Capex figure; tests retrieval of a specific capital allocation line item buried below the earnings headline.",
    ),
    EvalQuestion(
        id="XOM-2023-7-003",
        question="How did ExxonMobil's 2023 earnings compare to its record 2022 earnings, and what factors drove the change?",
        ticker="XOM",
        year=2023,
        section="Item 7",
        answer_keywords=["55,740", "36,010", "2022", "2023", "decreased"],
        question_type="comparative",
        difficulty="hard",
        notes="Hard: requires both the 2022 record figure and the 2023 figure plus the causal factors from the MD&A narrative; a single-chunk retrieval is unlikely to contain all three elements.",
    ),
    EvalQuestion(
        id="XOM-2024-1A-001",
        question="What commodity price risk does ExxonMobil identify as most material to its business in the 2024 risk factors?",
        ticker="XOM",
        year=2024,
        section="Item 1A",
        answer_keywords=["crude oil", "natural gas", "commodity", "price", "volatility"],
        question_type="factual",
        difficulty="medium",
        notes="Primary market risk for an integrated energy major; tests section-level routing to 1A rather than the Item 7 price-sensitivity discussion.",
    ),
    EvalQuestion(
        id="XOM-2024-1A-002",
        question="What major acquisition does ExxonMobil reference in its 2024 risk factors that expanded its Permian Basin operations?",
        ticker="XOM",
        year=2024,
        section="Item 1A",
        answer_keywords=["pioneer", "permian", "acquisition", "2024"],
        question_type="factual",
        difficulty="hard",
        notes="Hard: Pioneer Natural Resources acquisition (closed May 2024) appears in risk-factor language; tests whether the system retrieves a strategic corporate event from a risk section rather than an MD&A narrative.",
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # GOOGL  –  FY2023 · FY2024
    # ══════════════════════════════════════════════════════════════════════════

    EvalQuestion(
        id="GOOGL-2023-1A-001",
        question="What advertising revenue concentration risk does Alphabet identify in its 2023 risk factors?",
        ticker="GOOGL",
        year=2023,
        section="Item 1A",
        answer_keywords=["advertising", "revenues", "75%", "economic conditions"],
        question_type="factual",
        difficulty="medium",
        notes="Core business-model risk; tests retrieval of the specific advertising-dependence paragraph from a very long 1A section.",
    ),
    EvalQuestion(
        id="GOOGL-2023-1A-002",
        question="What generative AI competitive threat does Alphabet specifically describe in its 2023 risk factors related to search?",
        ticker="GOOGL",
        year=2023,
        section="Item 1A",
        answer_keywords=["AI", "search", "generative", "competition"],
        question_type="factual",
        difficulty="hard",
        notes="Hard: generative AI competitive risk was newly prominent in the 2023 filing; tests retrieval of a specific AI risk paragraph distinct from generic technology risk language.",
    ),
    EvalQuestion(
        id="GOOGL-2024-7-001",
        question="What were Alphabet's total revenues for fiscal year 2024?",
        ticker="GOOGL",
        year=2024,
        section="Item 7",
        answer_keywords=["350,018", "total revenues", "2024"],
        question_type="numerical",
        difficulty="easy",
        notes="Top-line revenue; tests year-tagging accuracy — the 2023 filing is much more widely indexed and may be substituted incorrectly.",
    ),
    EvalQuestion(
        id="GOOGL-2024-7-002",
        question="How did Google Cloud segment revenue change between fiscal year 2023 and fiscal year 2024?",
        ticker="GOOGL",
        year=2024,
        section="Item 7",
        answer_keywords=["google cloud", "43,229", "33,088", "revenues", "increased"],
        question_type="comparative",
        difficulty="medium",
        notes="Cloud segment two-year comparison; the Google Cloud row is adjacent to Google Services in the segment revenue table.",
    ),
    EvalQuestion(
        id="GOOGL-2024-7-003",
        question="What were YouTube advertising revenues for fiscal year 2024?",
        ticker="GOOGL",
        year=2024,
        section="Item 7",
        answer_keywords=["youtube ads", "36,147", "revenues", "2024"],
        question_type="numerical",
        difficulty="medium",
        notes="Sub-segment retrieval; the YouTube row sits directly between Google Search and Google Network in the revenue table — precision chunking test.",
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # BAC  –  FY2022 · FY2023
    # ══════════════════════════════════════════════════════════════════════════

    EvalQuestion(
        id="BAC-2022-7-001",
        question="How does Bank of America characterize its net interest income sensitivity to interest rate changes in its 2022 quantitative market risk disclosures?",
        ticker="BAC",
        year=2022,
        section="Item 7",
        answer_keywords=["net interest income", "interest rate", "sensitivity", "basis points"],
        question_type="factual",
        difficulty="medium",
        notes="Content lives in Item 7 MD&A (Market Risk Management / Interest Rate Risk Management for the Banking Book, Table 46); Item 7A itself is only a cross-reference to page 75.",
    ),
    EvalQuestion(
        id="BAC-2022-7-002",
        question="What was the estimated change in Bank of America's net interest income from an instantaneous parallel rate increase of 100 basis points, as disclosed in the 2022 filing?",
        ticker="BAC",
        year=2022,
        section="Item 7",
        answer_keywords=["100 bps", "net interest income", "parallel", "instantaneous"],
        question_type="numerical",
        difficulty="hard",
        notes="Hard: Table 46 'Estimated Banking Book NII Sensitivity to Curve Changes'; +100 bps parallel instantaneous shift → $3,829M for 2022. Section is Item 7 MD&A — Item 7A is only a redirect.",
    ),
    EvalQuestion(
        id="BAC-2023-7-001",
        question="What was Bank of America's net interest income for fiscal year 2023?",
        ticker="BAC",
        year=2023,
        section="Item 7",
        answer_keywords=["net interest income", "56,931", "2023"],
        question_type="numerical",
        difficulty="medium",
        notes="NII is the primary revenue driver for BAC; tests retrieval of a key banking metric from a dense financial results section.",
    ),
    EvalQuestion(
        id="BAC-2023-7-002",
        question="How did Bank of America's total deposits change between year-end 2022 and year-end 2023?",
        ticker="BAC",
        year=2023,
        section="Item 7",
        answer_keywords=["deposits", "2023", "2022", "decreased"],
        question_type="comparative",
        difficulty="medium",
        notes="Deposit outflow context (post-SVB rate environment); tests balance-sheet metric retrieval distinct from income-statement figures.",
    ),
    EvalQuestion(
        id="BAC-2023-7-003",
        question="What was Bank of America's provision for credit losses for fiscal year 2023?",
        ticker="BAC",
        year=2023,
        section="Item 7",
        answer_keywords=["provision for credit losses", "4,394", "2023"],
        question_type="numerical",
        difficulty="hard",
        notes="Hard: provision is a specific income-statement line item located below headline NII and noninterest income; tests fine-grained financial retrieval.",
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # PFE  –  FY2023 · FY2024
    # (No Item 1B, Item 4, or Item 9B in any PFE year.)
    # ══════════════════════════════════════════════════════════════════════════

    EvalQuestion(
        id="PFE-2023-1-001",
        question="What two COVID-19 products does Pfizer identify as experiencing significant revenue declines in its 2023 annual report?",
        ticker="PFE",
        year=2023,
        section="Item 1",
        answer_keywords=["Comirnaty", "Paxlovid", "COVID-19", "revenues"],
        question_type="factual",
        difficulty="medium",
        notes="COVID revenue cliff is central to the 2023 PFE narrative; tests whether the system retrieves the specific product-name language. Brand names capitalised as in filing.",
    ),
    EvalQuestion(
        id="PFE-2023-1-002",
        question="What major oncology-focused acquisition did Pfizer complete in 2023, and how is this transaction described in the business overview?",
        ticker="PFE",
        year=2023,
        section="Item 1",
        answer_keywords=["Seagen", "oncology", "acquisition", "December 14, 2023"],
        question_type="factual",
        difficulty="medium",
        notes="Seagen acquisition ($43.4B, closed Dec 14 2023) is the defining strategic event in the 2023 filing; 'December 14, 2023' is verbatim in the 10-K.",
    ),
    EvalQuestion(
        id="PFE-2023-7-002",
        question="What were Pfizer's total revenues for fiscal year 2023, and by what percentage did total revenues decrease compared to fiscal year 2022?",
        ticker="PFE",
        year=2023,
        section="Item 7",
        answer_keywords=["58,496", "revenues", "decreased", "42%", "2022"],
        question_type="numerical",
        difficulty="medium",
        notes="Verbatim from 2023 10-K Item 7 MD&A; $58,496M total revenues, 42% decrease from the 2022 COVID peak. Keywords confirmed from actual filing text.",
    ),
    EvalQuestion(
        id="PFE-2024-7-002",
        question="How did Pfizer's total revenues trend from the 2022 COVID-era peak through 2024, and which products drove the decline?",
        ticker="PFE",
        year=2024,
        section="Item 7",
        answer_keywords=["Comirnaty", "Paxlovid", "revenues", "2022", "increased", "7%"],
        question_type="trend",
        difficulty="hard",
        notes="Hard: multi-year revenue arc; 2022 peak $101.2B → 2023 $59.6B (recast) → 2024 $63.6B (+7%). 2023 comparison figure was recast in the 2024 10-K due to royalty income reclassification, so 2023 10-K figure of 58,496 does not appear here.",
    ),
    EvalQuestion(
        id="PFE-2023-7-001",
        question="What five strategic priorities did Pfizer enumerate in its 2023 Form 10-K for execution in fiscal year 2024?",
        ticker="PFE",
        year=2023,
        section="Item 7",
        answer_keywords=[
            "Achieve world-class oncology leadership",
            "Deliver next wave of pipeline innovation",
            "Maximize performance of our new products",
            "Expand margins by realigning our cost base",
            "Allocate capital to enhance shareholder value",
        ],
        question_type="factual",
        difficulty="hard",
        notes="Hard: all four bullet-point phrases are verbatim from 2023 10-K Item 7 MD&A. A retrieved chunk must contain every phrase; tests whether the chunker preserves the enumerated list intact.",
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # HD  –  FY2022 (ended Jan 30 2022) · FY2023 (ended Jan 28 2024)
    # ══════════════════════════════════════════════════════════════════════════

    EvalQuestion(
        id="HD-2022-1-001",
        question="How many Home Depot stores were in operation at the end of fiscal year 2022?",
        ticker="HD",
        year=2023,  # FY2022 ended Jan 29 2023 → filing hd-20230129 → corpus year=2023
        section="Item 1",
        answer_keywords=["2,322", "stores", "fiscal 2022"],
        question_type="numerical",
        difficulty="easy",
        notes="Store count from Item 1 of hd-20230129 (FY2022). HD's Jan fiscal year-end means FY2022 content lives under corpus year=2023.",
    ),
    EvalQuestion(
        id="HD-2022-1-002",
        question="What customer segments does Home Depot serve according to its fiscal year 2022 business description?",
        ticker="HD",
        year=2023,  # FY2022 ended Jan 29 2023 → filing hd-20230129 → corpus year=2023
        section="Item 1",
        answer_keywords=["DIY", "DIFM", "professional", "customers"],
        question_type="factual",
        difficulty="easy",
        notes="Customer segmentation from Item 1 of hd-20230129 (FY2022). HD's Jan fiscal year-end means FY2022 content lives under corpus year=2023.",
    ),
    EvalQuestion(
        id="HD-2023-7-001",
        question="What was Home Depot's total net sales for fiscal year 2023?",
        ticker="HD",
        year=2024,  # FY2023 ended Jan 28 2024 → filing hd-20240128 → corpus year=2024
        section="Item 7",
        answer_keywords=["152,669", "net sales", "fiscal 2023"],
        question_type="numerical",
        difficulty="easy",
        notes="Top-line revenue from hd-20240128 (FY2023). HD's Jan fiscal year-end means FY2023 content lives under corpus year=2024.",
    ),
    EvalQuestion(
        id="HD-2023-7-002",
        question="How did Home Depot's comparable store sales change in fiscal year 2023 compared to fiscal year 2022?",
        ticker="HD",
        year=2024,  # FY2023 ended Jan 28 2024 → filing hd-20240128 → corpus year=2024
        section="Item 7",
        answer_keywords=["comparable sales", "decreased", "(3.2)%", "fiscal 2023"],
        question_type="comparative",
        difficulty="medium",
        notes="Comp sales turned negative in FY2023; content in hd-20240128 under corpus year=2024.",
    ),
    EvalQuestion(
        id="HD-2023-7-003",
        question="What were Home Depot's diluted earnings per share for fiscal years 2022 and 2023, and in which direction did EPS move?",
        ticker="HD",
        year=2024,  # FY2023 ended Jan 28 2024 → filing hd-20240128 → corpus year=2024
        section="Item 7",
        answer_keywords=["diluted earnings per share", "15.11", "16.69", "decreased"],
        question_type="comparative",
        difficulty="hard",
        notes="Hard: two specific per-share figures required; content in hd-20240128 under corpus year=2024.",
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION-TARGETED CONCEPTUAL QUESTIONS (added to expose FM-1 / FM-3)
    #
    # These 15 questions specifically target Item 1A, 1B, 1C, and Item 1
    # sections with PARAPHRASTIC phrasing — the question text does not contain
    # the section name, so keyword matching across section boundaries cannot
    # substitute for correct section retrieval. They are designed to stress the
    # RRF intersection gate (FM-1) and the sparse-exclusion filter (FM-3).
    #
    # Difficulty is marked "hard" when the question is paraphrastic enough that
    # a dense retriever without section signal must genuinely distinguish between
    # Item 1A and Item 7 language about the same topic (e.g. risk description vs.
    # financial results).
    # ══════════════════════════════════════════════════════════════════════════

    # ── Item 1A  (risk factors — paraphrastic phrasing) ───────────────────────

    EvalQuestion(
        id="MSFT-2024-1A-002",
        question="What flaws or failure modes does Microsoft identify for its AI algorithms and training data?",
        ticker="MSFT",
        year=2024,
        section="Item 1A",
        answer_keywords=["AI", "algorithms", "flawed", "liability"],
        question_type="factual",
        difficulty="hard",
        notes="Paraphrastic: probe confirmed exact text 'AI algorithms or training methodologies may be flawed' in MSFT 2024 Item 1A. Dense retriever must beat Item 7 AI-revenue chunks on a question that says nothing about risk factors.",
    ),
    EvalQuestion(
        id="JNJ-2024-1A-001",
        question="What competitive threats from cheaper drug alternatives does Johnson & Johnson acknowledge could reduce its product revenues?",
        ticker="JNJ",
        year=2024,
        section="Item 1A",
        answer_keywords=["biosimilar", "generic", "patent", "revenues"],
        question_type="factual",
        difficulty="hard",
        notes="Paraphrastic: 'cheaper drug alternatives' must map to biosimilar/generic competition language in Item 1A. Tests section routing vs. Item 7 revenue discussion.",
    ),
    EvalQuestion(
        id="PFE-2024-1A-001",
        question="How does Pfizer describe the risk that loss of patent protection could allow competitors to undercut its drug prices?",
        ticker="PFE",
        year=2024,
        section="Item 1A",
        answer_keywords=["patent", "exclusivity", "generic", "competition"],
        question_type="factual",
        difficulty="hard",
        notes="Paraphrastic: patent cliff risk is in Item 1A; financials about it land in Item 7. Tests whether the system routes 'loss of patent protection' to risk factors rather than the revenue decline discussion.",
    ),
    EvalQuestion(
        id="GOOGL-2024-1A-001",
        question="What government investigations into anti-competitive conduct does Alphabet disclose in its 2024 annual report?",
        ticker="GOOGL",
        year=2024,
        section="Item 1A",
        answer_keywords=["antitrust", "competition", "DOJ", "regulation"],
        question_type="factual",
        difficulty="hard",
        notes="Paraphrastic: 'anti-competitive conduct' and 'government investigations' vs. 'antitrust' and 'DOJ' in filing. Item 1A carries the risk narrative; Item 3 carries legal proceedings. Tests cross-section disambiguation.",
    ),
    EvalQuestion(
        id="AAPL-2024-1A-001",
        question="What manufacturing and component sourcing vulnerabilities does Apple acknowledge in its 2024 annual report?",
        ticker="AAPL",
        year=2024,
        section="Item 1A",
        answer_keywords=["single source", "supply", "component", "manufacturing"],
        question_type="factual",
        difficulty="hard",
        notes="Paraphrastic: 'sourcing vulnerabilities' must map to single-source supplier / supply chain concentration language in Item 1A, not Item 1 product descriptions.",
    ),

    # ── Item 1C  (cybersecurity — paraphrastic phrasing) ──────────────────────

    EvalQuestion(
        id="MSFT-2024-1C-001",
        question="How does Microsoft describe board-level and executive oversight of cybersecurity risk in its 2024 annual report?",
        ticker="MSFT",
        year=2024,
        section="Item 1C",
        answer_keywords=["Board", "CISO", "Secure Future Initiative", "cybersecurity"],
        question_type="factual",
        difficulty="hard",
        notes="Paraphrastic: 'board-level oversight' maps to Board of Directors / Cybersecurity Governance Council language in Item 1C. From probe: chunk c5 (Item 1C) contains CISO + Secure Future Initiative + Board. Must beat Item 1A cybersecurity risk language.",
    ),
    EvalQuestion(
        id="PFE-2024-1C-001",
        question="What committee and executive role does Pfizer assign responsibility for cybersecurity risk management?",
        ticker="PFE",
        year=2024,
        section="Item 1C",
        answer_keywords=["CISO", "Audit Committee", "cybersecurity"],
        question_type="factual",
        difficulty="medium",
        notes="From probe: PFE 2024 Item 1C chunk contains 'CISO' + 'Audit Committee'. Paraphrastic: 'committee and executive role' vs. the named entities in the text.",
    ),
    EvalQuestion(
        id="JNJ-2024-1C-001",
        question="How does Johnson & Johnson govern information security risk at the board and management level?",
        ticker="JNJ",
        year=2024,
        section="Item 1C",
        answer_keywords=["CISO", "cybersecurity", "information security"],
        question_type="factual",
        difficulty="medium",
        notes="Paraphrastic phrasing of Item 1C disclosure. ⚠ Keywords estimated from standard 10-K Item 1C disclosures — verify against actual JNJ 2024 filing text.",
    ),

    # ── Item 1B  (unresolved staff comments — sparse section probe) ───────────

    EvalQuestion(
        id="MSFT-2024-1B-001",
        question="Does Microsoft have any outstanding unresolved comments from SEC staff regarding its annual reports?",
        ticker="MSFT",
        year=2024,
        section="Item 1B",
        answer_keywords=["None"],
        question_type="factual",
        difficulty="easy",
        notes="FM-3 probe: Item 1B is a sparse section ('None.') that is filtered out by exclude_sparse=True. This question will always fail while the sparse filter is active — it is an intentional canary for that failure mode.",
    ),
    EvalQuestion(
        id="WMT-2024-1B-001",
        question="Does Walmart disclose any unresolved SEC staff comments in its fiscal year 2024 annual report?",
        ticker="WMT",
        year=2024,
        section="Item 1B",
        answer_keywords=["None"],
        question_type="factual",
        difficulty="easy",
        notes="FM-3 probe: same sparse-section canary as MSFT-2024-1B-001. Expect 0% section_hit with exclude_sparse=True and 100% section_hit with --include-sparse.",
    ),

    # ── Item 1  (business overview — paraphrastic phrasing) ───────────────────

    EvalQuestion(
        id="JPM-2024-1-001",
        question="What principal lines of business does JPMorgan Chase operate according to its 2024 business description?",
        ticker="JPM",
        year=2024,
        section="Item 1",
        answer_keywords=["Consumer & Community Banking", "Commercial Banking", "Corporate & Investment Bank"],
        question_type="factual",
        difficulty="medium",
        notes="Paraphrastic: 'lines of business' maps to named segments in Item 1. Tests section routing vs. Item 7 segment results discussion that uses identical segment names.",
    ),
    EvalQuestion(
        id="XOM-2024-1-001",
        question="What major strategic acquisition completed in 2024 does ExxonMobil describe in its business overview, and what asset does it add?",
        ticker="XOM",
        year=2024,
        section="Item 1",
        answer_keywords=["Pioneer", "Permian", "acquisition"],
        question_type="factual",
        difficulty="medium",
        notes="Pioneer Natural Resources acquisition (closed May 2024) appears in both Item 1 and Item 1A. Tests whether Item 1 business-overview framing retrieves the Item 1 chunk rather than the Item 1A risk-factor framing.",
    ),
    EvalQuestion(
        id="WMT-2024-1-001",
        question="What are the primary retail segments that Walmart operates and where does it conduct business internationally?",
        ticker="WMT",
        year=2024,
        section="Item 1",
        answer_keywords=["Walmart U.S.", "Sam's Club", "Walmart International"],
        question_type="factual",
        difficulty="medium",
        notes="Paraphrastic: 'primary retail segments' maps to the named segment descriptions in Item 1. Item 7 also discusses segments by name — tests retrieval routing.",
    ),
    EvalQuestion(
        id="MSFT-2024-1-001",
        question="What AI assistant and platform products does Microsoft describe as new offerings in its 2024 business overview?",
        ticker="MSFT",
        year=2024,
        section="Item 1",
        answer_keywords=["Copilot", "Azure", "AI"],
        question_type="factual",
        difficulty="medium",
        notes="Paraphrastic: 'AI assistant' maps to Copilot; 'platform products' maps to Azure/cloud. Item 1 business description vs. Item 1A risk discussion of AI — same entities, different framing.",
    ),
    EvalQuestion(
        id="AAPL-2024-1-001",
        question="What subscription and digital services does Apple describe in its 2024 business overview?",
        ticker="AAPL",
        year=2024,
        section="Item 1",
        answer_keywords=["App Store", "Apple Music", "iCloud", "services"],
        question_type="factual",
        difficulty="medium",
        notes="Paraphrastic: 'subscription and digital services' maps to Apple's named Services segment offerings. Tests Item 1 business description vs. Item 7 Services revenue discussion.",
    ),
]


# ── Distribution stats ─────────────────────────────────────────────────────────
# Original 50 questions (financial/numerical) + 15 section-targeted conceptual
# questions added to expose FM-1 (RRF intersection gate) and FM-3 (sparse
# exclusion). The conceptual slice is 5×Item 1A + 3×Item 1C + 2×Item 1B + 5×Item 1.
STATS = {
    "total": len(QUESTIONS),
    "by_ticker": {
        "AAPL":  7,   # +2 conceptual (1A, Item 1)
        "MSFT":  8,   # +3 conceptual (1A, 1C, 1B, Item 1)
        "JPM":   6,   # +1 conceptual (Item 1)
        "JNJ":   7,   # +2 conceptual (1A, 1C)
        "WMT":   7,   # +2 conceptual (1B, Item 1)
        "XOM":   6,   # +1 conceptual (Item 1)
        "GOOGL": 6,   # +1 conceptual (1A)
        "BAC":   5,   # unchanged
        "PFE":   7,   # +2 conceptual (1A, 1C)
        "HD":    5,   # unchanged
    },
    "by_type": {
        "numerical":   21,
        "comparative": 11,
        "factual":     30,   # +15 conceptual questions
        "trend":        3,
    },
    "by_difficulty": {
        "easy":   16,   # +2 (Item 1B canaries)
        "medium": 30,   # +7 (Item 1 + some 1C)
        "hard":   19,   # +6 (paraphrastic 1A + 1C questions)
    },
    "by_section_conceptual": {
        "Item 1A": 5,
        "Item 1C": 3,
        "Item 1B": 2,
        "Item 1":  5,
    },
}


# ── Quick self-check (run as __main__) ────────────────────────────────────────
if __name__ == "__main__":
    from collections import Counter

    assert len(QUESTIONS) == 65, f"Expected 65 questions, got {len(QUESTIONS)}"

    tickers   = Counter(q.ticker for q in QUESTIONS)
    types     = Counter(q.question_type for q in QUESTIONS)
    diffs     = Counter(q.difficulty for q in QUESTIONS)

    # 5 per ticker
    for t, count in tickers.items():
        assert count == 5, f"{t}: expected 5 questions, got {count}"

    # Each ticker spans ≥ 2 years
    from itertools import groupby
    for ticker in set(q.ticker for q in QUESTIONS):
        years = set(q.year for q in QUESTIONS if q.ticker == ticker)
        assert len(years) >= 2, f"{ticker} only covers years: {years}"

    # Distribution requirements
    assert types["numerical"]   >= 15, f"Only {types['numerical']} numerical questions (need ≥15)"
    assert types["comparative"] >= 10, f"Only {types['comparative']} comparative questions (need ≥10)"
    assert diffs["hard"]        >= 8,  f"Only {diffs['hard']} hard questions (need ≥8)"

    # No absent sections
    for q in QUESTIONS:
        if q.ticker == "PFE":
            assert q.section not in ("Item 1B", "Item 4", "Item 9B"), \
                f"PFE question {q.id} targets absent section {q.section}"
        if q.ticker == "XOM":
            assert q.section != "Item 6", \
                f"XOM question {q.id} targets absent section Item 6"

    print("✓ All assertions passed")
    print(f"  Total questions : {len(QUESTIONS)}")
    print(f"  By type         : {dict(types)}")
    print(f"  By difficulty   : {dict(diffs)}")
    print(f"  By ticker       : {dict(tickers)}")

"""LangChain agent over the v4 retrieval stack, backed by Groq.

The agent's only job is decomposition and grounding: split a question into
single-filing retrievals, then answer strictly from what came back.
"""

import logging
import os

from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from agent.tools import CALL_LOG, list_corpus, reset_call_log, search_10k

log = logging.getLogger(__name__)

# llama-3.1-70b-versatile was retired from Groq's production models; 3.3 is the
# current 70B versatile model. Override with GROQ_MODEL if you want another.
DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

SYSTEM_PROMPT = """You are a financial research assistant with access to a \
private corpus of SEC 10-K filings.

CORPUS SCOPE — this is all you can see:
- Companies: AAPL (Apple), MSFT (Microsoft), GOOGL (Alphabet/Google), \
JPM (JPMorgan Chase), BAC (Bank of America), XOM (ExxonMobil), WMT (Walmart), \
JNJ (Johnson & Johnson), PFE (Pfizer), HD (Home Depot)
- Fiscal years: 2022, 2023 and 2024 ONLY

If a question asks about a company or year outside that scope, say so plainly \
instead of answering from general knowledge.

HOW TO USE THE TOOLS:
list_corpus tells you exactly which companies, fiscal years and filings exist, \
with chunk and section counts. Call it first when the question is vague about \
scope ("healthcare companies", "the banks", "the latest filings") or when you \
need to confirm something exists before searching. It also states how each \
filer labels its fiscal years, which matters because some labels do not match \
the year the period ends in.


The search_10k tool searches ONE company and ONE fiscal year at a time. It \
cannot compare or aggregate. So:
- Multi-year trend question ("revenue from 2022 to 2024") -> call the tool once \
per year, then compare the figures yourself.
- Multi-company comparison ("Microsoft and J&J R&D spend") -> call the tool once \
per company.
- Single company and year -> one call is enough.
Pass ticker and year explicitly whenever you know them. If the question does not \
name a company, leave ticker empty and let the query text drive the search.

USING THE SECTION PARAMETER:
The section parameter hard-pins retrieval to a specific part of the 10-K. Use it \
when the question is unambiguously about one section — it prevents unrelated \
sections from crowding out the right content. The valid values are:
  '1'   — Item 1: Business (company overview, products, strategy)
  '1A'  — Item 1A: Risk Factors (risks, litigation, regulatory exposure)
  '1B'  — Item 1B: Unresolved Staff Comments
  '7'   — Item 7: MD&A (management's discussion, revenue narrative, outlook)
  '8'   — Item 8: Financial Statements (income statement, balance sheet, cash flow)
  '9A'  — Item 9A: Controls and Procedures
Examples: "risk factors" or "what risks does X face" -> section='1A'. \
"revenue" or "earnings" or "net income" -> do NOT pin a section (the figure may \
be in Item 7 narrative or Item 8 tables; let retrieval decide). \
"business overview" or "what does X do" -> section='1'. \
When in doubt, omit section and let the retriever rank freely.

CHOOSING A FISCAL YEAR:
If the user's query does not specify a fiscal year, default to the most recent \
available year (2024) and explicitly state that in your answer (e.g. "Based on \
Pfizer's FY2024 10-K..."). Only search multiple years if the query explicitly \
implies a trend, comparison, or change over time — look for words like "trend", \
"changed", "grew", "declined", "over time", "compare", "from X to Y", "history". \
In that case, make one tool call per year needed. Never silently pick a year \
without stating it in the answer.

GROUNDING RULES — these are absolute:
- NEVER state a number, date, or fact that does not appear verbatim in a \
retrieved chunk. Do not compute, estimate, or recall figures from memory.
- Chunks below the relevance floor are filtered out before you see them. If a \
tool call reports no confident matches, say so for that company or year and do \
NOT substitute general knowledge.
- If the retrieved chunks do not answer the question, say exactly what is \
missing. An honest "the filings retrieved do not state this" is correct; a \
plausible guess is a failure.
- Percentages and growth rates: only state them if they appear in a chunk, or if \
you are computing directly from two figures that each appear in chunks — and say \
which figures you used.

ANSWER FORMAT — always both parts:
1. A prose answer, grounded in the chunks.
2. A "Sources:" section listing every chunk you actually used, one per line:
   [TICKER | FY{year} | {section} | chunk {index}] "{brief excerpt}"
Only cite chunks you actually drew on. Do not pad the list."""


def build_agent(model=None, temperature=0):
    """Construct the agent. Requires GROQ_API_KEY in the environment."""
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
            "and export it, or put it in sec-rag/.env")

    llm = ChatGroq(
        model=model or DEFAULT_MODEL,
        temperature=temperature,
        # A trend question needs 3 sequential tool calls plus a synthesis turn;
        # a short timeout truncates the run mid-plan.
        timeout=120,
        max_retries=2,
    )
    return create_react_agent(llm, [search_10k, list_corpus], prompt=SYSTEM_PROMPT)


# llama on Groq intermittently emits "<function=name{...}</function>" instead of
# a real tool call, which Groq rejects with a 400 tool_use_failed. It is a
# generation fluke, so a retry usually clears it; FALLBACK_MODEL is the escape
# hatch if it does not.
FALLBACK_MODEL = os.environ.get("GROQ_FALLBACK_MODEL", "qwen/qwen3.6-27b")


def run(question, model=None, temperature=0, recursion_limit=25, attempts=3):
    """Run one question. Returns (answer_text, tool_calls, messages)."""
    chosen = model or DEFAULT_MODEL
    last = None
    for attempt in range(1, attempts + 1):
        reset_call_log()
        try:
            agent = build_agent(model=chosen, temperature=temperature)
            result = agent.invoke(
                {"messages": [{"role": "user", "content": question}]},
                config={"recursion_limit": recursion_limit},
            )
            messages = result["messages"]
            answer = messages[-1].content if messages else ""
            return answer, list(CALL_LOG), messages
        except Exception as exc:
            last = exc
            if "tool_use_failed" not in str(exc) and "Failed to call a function" not in str(exc):
                raise
            log.warning("attempt %d/%d: %s emitted a malformed tool call; retrying",
                        attempt, attempts, chosen)
            if attempt == attempts - 1 and chosen != FALLBACK_MODEL:
                log.warning("falling back to %s", FALLBACK_MODEL)
                chosen = FALLBACK_MODEL
    raise last

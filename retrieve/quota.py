"""Per-section budgeting for the rerank candidate pool.

v4's own docstring names this failure as FM-2: "With a tight ticker+year filter,
the largest sections (Item 7/8) fill every slot and short sections get crowded
out." v4's answer was to widen the cut from 5 to 20 and let the cross-encoder
sort it out. That helps, but it does not change *which* chunks are in the pool,
and the pool is where the crowding happens.

The imbalance is structural, not incidental. Measured over the corpus:

    JNJ FY2023   Item 8 = 145 chunks   Item 1A =  19 chunks    7.6 : 1
    JPM FY2023   Item 8 = 402 chunks   Item 1A =  70 chunks    5.7 : 1
    BAC FY2024   Item 8 = 289 chunks   Item 1A =  56 chunks    5.2 : 1

A flat top-N over a filing therefore fills with Item 7/8 on volume alone, before
relevance is considered at all. verification_report.md caught it happening: the
MSFT risk-factors call returned one Item 1A chunk above the floor, and the JNJ
one put Item 7 at rank 1 and dropped four of five.

So: draw a deeper pool, then cap how much of it any single section may occupy.
Capping alone would shrink the pool, which is why the draw is widened first --
the cap reallocates slots, it does not discard them.
"""

import logging
from collections import defaultdict

import config

log = logging.getLogger(__name__)


def apply_section_quota(pool, n, max_share=None):
    """Select `n` results from `pool`, capping any one section's share.

    `pool` must already be sorted best-first; relative order is preserved both
    within a section and in the final result, so this only ever *removes*
    lower-ranked chunks of an over-represented section in favour of
    higher-ranked chunks of an under-represented one.

    Returns the pool untouched when it is already at or under `n` -- there is
    nothing to reallocate.
    """
    if not pool or len(pool) <= n:
        return pool[:n]

    share = config.SECTION_QUOTA_SHARE if max_share is None else max_share
    if share >= 1.0:
        return pool[:n]

    cap = max(1, int(n * share))

    by_section = defaultdict(list)
    for item in pool:
        by_section[item.section].append(item)

    # Nothing to do when the pool is already spread thin enough.
    if all(len(v) <= cap for v in by_section.values()):
        return pool[:n]

    kept, taken, overflow = [], defaultdict(int), []
    for item in pool:
        if taken[item.section] < cap:
            kept.append(item)
            taken[item.section] += 1
        else:
            overflow.append(item)
        if len(kept) == n:
            break

    # If capping left the pool short (few sections represented), top it back up
    # from the overflow in rank order rather than returning fewer than asked.
    if len(kept) < n:
        kept.extend(overflow[:n - len(kept)])
        kept.sort(key=lambda r: pool.index(r))

    if log.isEnabledFor(logging.DEBUG):
        before = {s: len(v) for s, v in by_section.items()}
        after = defaultdict(int)
        for item in kept:
            after[item.section] += 1
        log.debug("section quota (cap=%d of %d): %s -> %s",
                  cap, n, dict(before), dict(after))

    return kept[:n]

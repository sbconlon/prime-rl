"""Root allocation for the branched dataset (plan branched-dataset D-9).

Pure functions over trunk lengths and returns:

* stratified_roots -- B decision points spread over the trunk with a share of
  them in the first third (the early tilt); never the terminal turn.
* pivotal_roots    -- the two-stage rule: a candidate root is pivotal when its
  stage-1 probe branch returned something other than the trunk did.
"""

from __future__ import annotations


def _evenly_spaced(pool: list[int], n: int) -> list[int]:
    """n distinct entries of pool at evenly spaced positions (pool sorted)."""
    if n <= 0 or not pool:
        return []
    if n >= len(pool):
        return list(pool)
    if n == 1:
        return [pool[len(pool) // 2]]
    last = len(pool) - 1
    picks = {pool[round(i * last / (n - 1))] for i in range(n)}
    # rounding can collide; top up from the pool in order
    for x in pool:
        if len(picks) >= n:
            break
        picks.add(x)
    return sorted(picks)


def stratified_roots(T: int, B: int, early_tilt: float = 0.5) -> list[int]:
    """Decision-point indices to branch from along a trunk of T decision points.

    Candidates are 0..T-2 (the terminal turn T-1 is never a root). When there are
    at most B candidates, all are returned. Otherwise round(B * early_tilt) come
    from the first third of the candidates and the rest from the remainder, each
    evenly spaced; a pool too small to fill its share hands the slack to the other.
    """
    if T <= 1 or B <= 0:
        return []
    cands = list(range(T - 1))
    if len(cands) <= B:
        return cands
    cut = max(1, len(cands) // 3)
    early_pool, late_pool = cands[:cut], cands[cut:]
    n_early = min(int(round(B * early_tilt)), len(early_pool))
    n_late = min(B - n_early, len(late_pool))
    # slack: a short pool gives its unfilled slots to the other
    n_early = min(B - n_late, len(early_pool))
    picks = set(_evenly_spaced(early_pool, n_early)) | set(_evenly_spaced(late_pool, n_late))
    return sorted(picks)


def pivotal_roots(stage1: dict[int, float | None], trunk_return: float, B2: int) -> list[int]:
    """Two-stage allocation: candidates whose stage-1 branch return differs from the
    trunk's, in trunk order, capped at B2. A None return (dropped branch) is not
    pivotal."""
    out = [t for t in sorted(stage1) if stage1[t] is not None and stage1[t] != trunk_return]
    return out[: max(0, B2)]

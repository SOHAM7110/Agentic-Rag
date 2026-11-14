def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: score(d) = sum_r 1 / (k + rank_r(d))."""
    score: dict[int, float] = {}
    for r in rankings:
        for rank, i in enumerate(r, start=1):
            score[i] = score.get(i, 0.0) + 1.0 / (k + rank)
    return sorted(score.items(), key=lambda x: -x[1])

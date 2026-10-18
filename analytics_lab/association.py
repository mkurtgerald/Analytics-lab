"""Bounded, deterministic weighted bipartite assignment using only the stdlib."""
from __future__ import annotations

import math


def maximum_cardinality_weighted_matching(
    weights: tuple[tuple[float | None, ...], ...], *, max_size: int,
) -> tuple[tuple[int, int], ...]:
    """Maximize valid pair count, then the exact sum of supplied float weights.

    ``None`` denotes an invalid edge; finite weights in [0, 1] are valid edges,
    including zero. Both matrix dimensions are bounded by ``max_size`` before
    assignment. Equal objectives resolve deterministically in matrix order, so
    callers should supply their desired canonical row and column ordering.

    The rectangular Hungarian algorithm saturates the smaller dimension with
    zero-cost invalid edges allowed as placeholders. A valid edge's cardinality
    bonus exceeds every possible total weight, so filtering those placeholders
    yields a maximum-cardinality matching. Float ratios are scaled to exact
    integers first: adding the bonus cannot erase small IoU differences.
    Memory is O(rows * columns); assignment takes
    O(min(rows, columns)^2 * max(rows, columns)) operations.
    """
    if type(max_size) is not int or max_size < 1:
        raise ValueError("max_size must be a positive integer")
    if not isinstance(weights, tuple):
        raise ValueError("weights must be a tuple of row tuples")
    rows = len(weights)
    if rows > max_size:
        raise RuntimeError("assignment matrix exceeds configured capacity")
    if any(not isinstance(row, tuple) for row in weights):
        raise ValueError("weights must be a tuple of row tuples")
    columns = len(weights[0]) if rows else 0
    if columns > max_size:
        raise RuntimeError("assignment matrix exceeds configured capacity")
    if any(len(row) != columns for row in weights):
        raise ValueError("weights must be rectangular")

    ratios: list[list[tuple[int, int] | None]] = []
    scale = 1
    for row in weights:
        values: list[tuple[int, int] | None] = []
        for weight in row:
            if weight is None:
                values.append(None)
                continue
            if (type(weight) not in (int, float) or not 0 <= weight <= 1
                    or not math.isfinite(weight)):
                raise ValueError("assignment weights must be finite and in [0, 1]")
            ratio = float(weight).as_integer_ratio()
            values.append(ratio)
            # Every binary float denominator is a power of two.
            scale = max(scale, ratio[1])
        ratios.append(values)
    if not rows or not columns:
        return ()

    bonus = (min(rows, columns) + 1) * scale
    costs = [
        [0 if ratio is None else -(bonus + ratio[0] * (scale // ratio[1])) for ratio in row]
        for row in ratios
    ]
    transposed = rows > columns
    if transposed:
        costs = [list(row) for row in zip(*costs)]
    row_count, column_count = len(costs), len(costs[0])

    row_potential = [0] * (row_count + 1)
    column_potential = [0] * (column_count + 1)
    column_row = [0] * (column_count + 1)
    previous_column = [0] * (column_count + 1)
    for row in range(1, row_count + 1):
        column_row[0] = row
        column = 0
        minimum: list[int | None] = [None] * (column_count + 1)
        used = [False] * (column_count + 1)
        while True:
            used[column] = True
            current_row = column_row[column]
            delta: int | None = None
            next_column = 0
            for candidate in range(1, column_count + 1):
                if used[candidate]:
                    continue
                reduced = (costs[current_row - 1][candidate - 1]
                           - row_potential[current_row] - column_potential[candidate])
                if minimum[candidate] is None or reduced < minimum[candidate]:
                    minimum[candidate] = reduced
                    previous_column[candidate] = column
                if delta is None or minimum[candidate] < delta:
                    delta = minimum[candidate]
                    next_column = candidate
            assert delta is not None
            for candidate in range(column_count + 1):
                if used[candidate]:
                    row_potential[column_row[candidate]] += delta
                    column_potential[candidate] -= delta
                elif minimum[candidate] is not None:
                    minimum[candidate] -= delta
            column = next_column
            if column_row[column] == 0:
                break
        while column:
            prior = previous_column[column]
            column_row[column] = column_row[prior]
            column = prior

    matches: list[tuple[int, int]] = []
    for column in range(1, column_count + 1):
        if column_row[column]:
            row_index, column_index = column_row[column] - 1, column - 1
            if transposed:
                row_index, column_index = column_index, row_index
            if weights[row_index][column_index] is not None:
                matches.append((row_index, column_index))
    return tuple(sorted(matches))

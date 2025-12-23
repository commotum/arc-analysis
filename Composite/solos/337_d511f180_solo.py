# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "d511f180"
SERIAL = "337"
URL    = "https://arcprize.org/play?task=d511f180"

# --- Code Golf Concepts ---
CONCEPTS = [
    "associate_colors_to_colors",
]

# --- Example Grids ---
E1_IN = np.array([
    [2, 7, 8, 8, 8],
    [5, 5, 6, 5, 4],
    [8, 5, 5, 5, 2],
    [8, 8, 4, 3, 6],
    [6, 5, 1, 9, 3],
], dtype=int)

E1_OUT = np.array([
    [2, 7, 5, 5, 5],
    [8, 8, 6, 8, 4],
    [5, 8, 8, 8, 2],
    [5, 5, 4, 3, 6],
    [6, 8, 1, 9, 3],
], dtype=int)

E2_IN = np.array([
    [3, 5, 1],
    [4, 5, 8],
    [2, 4, 9],
], dtype=int)

E2_OUT = np.array([
    [3, 8, 1],
    [4, 8, 5],
    [2, 4, 9],
], dtype=int)

E3_IN = np.array([
    [6, 5, 3],
    [5, 7, 5],
    [8, 8, 2],
], dtype=int)

E3_OUT = np.array([
    [6, 8, 3],
    [8, 7, 8],
    [5, 5, 2],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [8, 8, 4, 5],
    [3, 8, 7, 5],
    [3, 7, 1, 9],
    [6, 4, 8, 8],
], dtype=int)

T_OUT = np.array([
    [5, 5, 4, 8],
    [3, 5, 7, 8],
    [3, 7, 1, 9],
    [6, 4, 5, 5],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
    return [[A ^ 13 * (A in (5, 8)) for A in A] for A in j]


# --- Code Golf Solution (Compressed) ---
def q(g):
    return g * -1 and g ^ 84 % g % 3 * 13 or [*map(p, g)]


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, randint, sample, uniform

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

def difference(
    a: Container,
    b: Container
) -> Container:
    """ difference """
    return type(a)(e for e in a if e not in b)

def totuple(
    container: FrozenSet
) -> Tuple:
    """ conversion to tuple """
    return tuple(container)

def interval(
    start: Integer,
    stop: Integer,
    step: Integer
) -> Tuple:
    """ range """
    return tuple(range(start, stop, step))

def asindices(
    grid: Grid
) -> Indices:
    """ indices of all grid cells """
    return frozenset((i, j) for i in range(len(grid)) for j in range(len(grid[0])))

def toindices(
    patch: Patch
) -> Indices:
    """ indices of object cells """
    if len(patch) == 0:
        return frozenset()
    if isinstance(next(iter(patch))[1], tuple):
        return frozenset(index for value, index in patch)
    return patch

def fill(
    grid: Grid,
    value: Integer,
    patch: Patch
) -> Grid:
    """ fill value at indices """
    h, w = len(grid), len(grid[0])
    grid_filled = list(list(row) for row in grid)
    for i, j in toindices(patch):
        if 0 <= i < h and 0 <= j < w:
            grid_filled[i][j] = value
    return tuple(tuple(row) for row in grid_filled)

def canvas(
    value: Integer,
    dimensions: IntegerTuple
) -> Grid:
    """ grid construction """
    return tuple(tuple(value for j in range(dimensions[1])) for i in range(dimensions[0]))

rng = []

def unifint(
    diff_lb: float,
    diff_ub: float,
    bounds: Tuple[int, int]
) -> int:
    """
    diff_lb: lower bound for difficulty, must be in range [0, diff_ub]
    diff_ub: upper bound for difficulty, must be in range [diff_lb, 1]
    bounds: interval [a, b] determining the integer values that can be sampled
    """
    a, b = bounds
    d = uniform(diff_lb, diff_ub)
    global rng
    rng.append(d)
    return min(max(a, round(a + (b - a) * d)), b)

def generate_d511f180(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (5, 8))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    numc = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(cols, numc)
    c = canvas(-1, (h, w))
    inds = totuple(asindices(c))
    numbg = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    bginds = sample(inds, numbg)
    idx = randint(0, numbg)
    blues = bginds[:idx]
    greys = bginds[idx:]
    rem = difference(inds, bginds)
    gi = fill(c, 8, blues)
    gi = fill(gi, 5, greys)
    go = fill(c, 5, blues)
    go = fill(go, 8, greys)
    for ij in rem:
        col = choice(ccols)
        gi = fill(gi, col, {ij})
        go = fill(go, col, {ij})
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
FIVE = 5

EIGHT = 8

def switch(
    grid: Grid,
    a: Integer,
    b: Integer
) -> Grid:
    """ color switching """
    return tuple(tuple(v if (v != a and v != b) else {a: b, b: a}[v] for v in r) for r in grid)

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_d511f180(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = switch(I, FIVE, EIGHT)
    return x0


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_d511f180(inp)
        assert pred == _to_grid(expected), f"{name} failed"

# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "0d3d703e"
SERIAL = "016"
URL    = "https://arcprize.org/play?task=0d3d703e"

# --- Code Golf Concepts ---
CONCEPTS = [
    "associate_colors_to_colors",
]

# --- Example Grids ---
E1_IN = np.array([
    [3, 1, 2],
    [3, 1, 2],
    [3, 1, 2],
], dtype=int)

E1_OUT = np.array([
    [4, 5, 6],
    [4, 5, 6],
    [4, 5, 6],
], dtype=int)

E2_IN = np.array([
    [2, 3, 8],
    [2, 3, 8],
    [2, 3, 8],
], dtype=int)

E2_OUT = np.array([
    [6, 4, 9],
    [6, 4, 9],
    [6, 4, 9],
], dtype=int)

E3_IN = np.array([
    [5, 8, 6],
    [5, 8, 6],
    [5, 8, 6],
], dtype=int)

E3_OUT = np.array([
    [1, 9, 2],
    [1, 9, 2],
    [1, 9, 2],
], dtype=int)

E4_IN = np.array([
    [9, 4, 2],
    [9, 4, 2],
    [9, 4, 2],
], dtype=int)

E4_OUT = np.array([
    [8, 3, 6],
    [8, 3, 6],
    [8, 3, 6],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [8, 1, 3],
    [8, 1, 3],
    [8, 1, 3],
], dtype=int)

T_OUT = np.array([
    [9, 5, 4],
    [9, 5, 4],
    [9, 5, 4],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j, A=[0, 5, 6, 4, 3, 1, 2, 7, 9, 8]):
    return [[A[x] for x in r] for r in j]


# --- Code Golf Solution (Compressed) ---
def q(g):
    return [[x ** 10 % 95 % 18 ^ 4 for x in g[0]]] * 3


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, sample, uniform

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

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

def generate_0d3d703e(diff_lb: float, diff_ub: float) -> dict:
    incols = (1, 2, 3, 4, 5, 6, 8, 9)
    outcols = (5, 6, 4, 3, 1, 2, 9, 8)
    k = len(incols)
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    gi = canvas(-1, (h, w))
    go = canvas(-1, (h, w))
    inds = asindices(gi)
    numc = unifint(diff_lb, diff_ub, (1, k))
    idxes = sample(interval(0, k, 1), numc)
    for ij in inds:
        idx = choice(idxes)
        gi = fill(gi, incols[idx], {ij})
        go = fill(go, outcols[idx], {ij})
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
ONE = 1

TWO = 2

THREE = 3

FOUR = 4

FIVE = 5

SIX = 6

EIGHT = 8

NINE = 9

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

def verify_0d3d703e(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = switch(I, THREE, FOUR)
    x1 = switch(x0, EIGHT, NINE)
    x2 = switch(x1, TWO, SIX)
    x3 = switch(x2, ONE, FIVE)
    return x3


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("E4", E4_IN, E4_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_0d3d703e(inp)
        assert pred == _to_grid(expected), f"{name} failed"

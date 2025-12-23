# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "f76d97a5"
SERIAL = "389"
URL    = "https://arcprize.org/play?task=f76d97a5"

# --- Code Golf Concepts ---
CONCEPTS = [
    "take_negative",
    "recoloring",
    "associate_colors_to_colors",
]

# --- Example Grids ---
E1_IN = np.array([
    [4, 5, 4],
    [5, 5, 5],
    [4, 5, 4],
], dtype=int)

E1_OUT = np.array([
    [0, 4, 0],
    [4, 4, 4],
    [0, 4, 0],
], dtype=int)

E2_IN = np.array([
    [5, 5, 6, 6, 6],
    [6, 5, 5, 6, 6],
    [6, 6, 5, 5, 6],
    [6, 6, 6, 5, 5],
    [5, 6, 6, 6, 5],
], dtype=int)

E2_OUT = np.array([
    [6, 6, 0, 0, 0],
    [0, 6, 6, 0, 0],
    [0, 0, 6, 6, 0],
    [0, 0, 0, 6, 6],
    [6, 0, 0, 0, 6],
], dtype=int)

E3_IN = np.array([
    [9, 5, 9, 9, 9],
    [9, 9, 5, 5, 9],
    [9, 5, 9, 9, 9],
    [9, 9, 5, 9, 9],
    [9, 9, 9, 5, 5],
], dtype=int)

E3_OUT = np.array([
    [0, 9, 0, 0, 0],
    [0, 0, 9, 9, 0],
    [0, 9, 0, 0, 0],
    [0, 0, 9, 0, 0],
    [0, 0, 0, 9, 9],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [3, 3, 3, 5, 3],
    [3, 5, 3, 3, 3],
    [3, 5, 5, 3, 5],
    [3, 3, 3, 5, 3],
    [5, 5, 5, 3, 3],
], dtype=int)

T_OUT = np.array([
    [0, 0, 0, 3, 0],
    [0, 3, 0, 0, 0],
    [0, 3, 3, 0, 3],
    [0, 0, 0, 3, 0],
    [3, 3, 3, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):A=[i for s in j for i in s];A=[c for c in set(A)if c not in[0,5]][0];j=[[A if C==5 else 0 for C in R]for R in j];return j


# --- Code Golf Solution (Compressed) ---
def q(g):
    return [[sum({*sum(g, r)} - {5, x}) for x in r] for r in g]


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

def totuple(
    container: FrozenSet
) -> Tuple:
    """ conversion to tuple """
    return tuple(container)

def remove(
    value: Any,
    container: Container
) -> Container:
    """ remove item from container """
    return type(container)(e for e in container if e != value)

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

def generate_f76d97a5(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(0, remove(5, interval(0, 10, 1)))
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    col = choice(cols)
    gi = canvas(5, (h, w))
    go = canvas(col, (h, w))
    numdev = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    num = choice((numdev, h * w - numdev))
    num = min(max(1, num), h * w)
    inds = totuple(asindices(gi))
    locs = sample(inds, num)
    gi = fill(gi, col, locs)
    go = fill(go, 0, locs)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
IntegerSet = FrozenSet[Integer]

Element = Union[Object, Grid]

ZERO = 0

FIVE = 5

def first(
    container: Container
) -> Any:
    """ first item of container """
    return next(iter(container))

def ofcolor(
    grid: Grid,
    value: Integer
) -> Indices:
    """ indices of all grid cells with value """
    return frozenset((i, j) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value)

def palette(
    element: Element
) -> IntegerSet:
    """ colors occurring in object or grid """
    if isinstance(element, tuple):
        return frozenset({v for r in element for v in r})
    return frozenset({v for v, _ in element})

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_f76d97a5(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = palette(I)
    x1 = remove(FIVE, x0)
    x2 = first(x1)
    x3 = ofcolor(I, x2)
    x4 = fill(I, ZERO, x3)
    x5 = ofcolor(I, FIVE)
    x6 = fill(x4, x2, x5)
    return x6


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_f76d97a5(inp)
        assert pred == _to_grid(expected), f"{name} failed"

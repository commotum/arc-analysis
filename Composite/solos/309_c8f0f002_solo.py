# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "c8f0f002"
SERIAL = "309"
URL    = "https://arcprize.org/play?task=c8f0f002"

# --- Code Golf Concepts ---
CONCEPTS = [
    "recoloring",
    "associate_colors_to_colors",
]

# --- Example Grids ---
E1_IN = np.array([
    [1, 8, 8, 7, 7, 8],
    [1, 1, 7, 7, 1, 8],
    [7, 1, 1, 7, 7, 8],
], dtype=int)

E1_OUT = np.array([
    [1, 8, 8, 5, 5, 8],
    [1, 1, 5, 5, 1, 8],
    [5, 1, 1, 5, 5, 8],
], dtype=int)

E2_IN = np.array([
    [7, 7, 7, 1],
    [1, 8, 1, 7],
    [7, 1, 1, 7],
], dtype=int)

E2_OUT = np.array([
    [5, 5, 5, 1],
    [1, 8, 1, 5],
    [5, 1, 1, 5],
], dtype=int)

E3_IN = np.array([
    [1, 8, 1, 7, 1],
    [7, 8, 8, 1, 1],
    [7, 1, 8, 8, 7],
], dtype=int)

E3_OUT = np.array([
    [1, 8, 1, 5, 1],
    [5, 8, 8, 1, 1],
    [5, 1, 8, 8, 5],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [1, 7, 7, 1, 7],
    [8, 1, 7, 7, 7],
    [8, 7, 1, 7, 8],
], dtype=int)

T_OUT = np.array([
    [1, 5, 5, 1, 5],
    [8, 1, 5, 5, 5],
    [8, 5, 1, 5, 8],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
    return [[x - 2 * (x == 7) for x in r] for r in j]


# --- Code Golf Solution (Compressed) ---
def q(g):
    return g * -1 and g & -3 or [*map(p, g)]


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

def generate_c8f0f002(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(7, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    numc = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(cols, numc)
    c = canvas(-1, (h, w))
    inds = totuple(asindices(c))
    numo = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    orng = sample(inds, numo)
    rem = difference(inds, orng)
    gi = fill(c, 7, orng)
    go = fill(c, 5, orng)
    for ij in rem:
        col = choice(ccols)
        gi = fill(gi, col, {ij})
        go = fill(go, col, {ij})
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
FIVE = 5

SEVEN = 7

def replace(
    grid: Grid,
    replacee: Integer,
    replacer: Integer
) -> Grid:
    """ color substitution """
    return tuple(tuple(replacer if v == replacee else v for v in r) for r in grid)

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_c8f0f002(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = replace(I, SEVEN, FIVE)
    return x0


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_c8f0f002(inp)
        assert pred == _to_grid(expected), f"{name} failed"

# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "b1948b0a"
SERIAL = "276"
URL    = "https://arcprize.org/play?task=b1948b0a"

# --- Code Golf Concepts ---
CONCEPTS = [
    "recoloring",
    "associate_colors_to_colors",
]

# --- Example Grids ---
E1_IN = np.array([
    [6, 6, 7, 6],
    [6, 6, 7, 7],
    [7, 7, 6, 7],
], dtype=int)

E1_OUT = np.array([
    [2, 2, 7, 2],
    [2, 2, 7, 7],
    [7, 7, 2, 7],
], dtype=int)

E2_IN = np.array([
    [7, 7, 7, 6],
    [6, 6, 7, 6],
    [7, 7, 6, 7],
    [7, 6, 7, 7],
    [7, 6, 7, 6],
    [6, 6, 6, 7],
], dtype=int)

E2_OUT = np.array([
    [7, 7, 7, 2],
    [2, 2, 7, 2],
    [7, 7, 2, 7],
    [7, 2, 7, 7],
    [7, 2, 7, 2],
    [2, 2, 2, 7],
], dtype=int)

E3_IN = np.array([
    [7, 7, 6, 6, 6, 6],
    [6, 7, 6, 7, 7, 7],
    [7, 6, 7, 7, 6, 7],
], dtype=int)

E3_OUT = np.array([
    [7, 7, 2, 2, 2, 2],
    [2, 7, 2, 7, 7, 7],
    [7, 2, 7, 7, 2, 7],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [6, 7, 7, 6],
    [6, 7, 6, 7],
    [7, 7, 7, 6],
    [7, 6, 7, 6],
], dtype=int)

T_OUT = np.array([
    [2, 7, 7, 2],
    [2, 7, 2, 7],
    [7, 7, 7, 2],
    [7, 2, 7, 2],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(g):
    return [[{6: 2, 7: 7}.get(x, x) for x in r] for r in g]


# --- Code Golf Solution (Compressed) ---
def q(g):
    return g * -1 and -g % 6 | 2 or [*map(p, g)]


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

def generate_b1948b0a(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(6, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    npd = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    np = choice((npd, h * w - npd))
    np = min(max(0, npd), h * w)
    gi = canvas(6, (h, w))
    inds = totuple(asindices(gi))
    pp = sample(inds, np)
    npp = difference(inds, pp)
    for ij in npp:
        gi = fill(gi, choice(cols), {ij})
    go = fill(gi, 2, pp)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
TWO = 2

SIX = 6

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

def verify_b1948b0a(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = replace(I, SIX, TWO)
    return x0


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_b1948b0a(inp)
        assert pred == _to_grid(expected), f"{name} failed"

# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "d631b094"
SERIAL = "339"
URL    = "https://arcprize.org/play?task=d631b094"

# --- Code Golf Concepts ---
CONCEPTS = [
    "count_tiles",
    "dominant_color",
    "summarize",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
], dtype=int)

E1_OUT = np.array([
    [1, 1],
], dtype=int)

E2_IN = np.array([
    [0, 2, 0],
    [2, 0, 0],
    [0, 2, 0],
], dtype=int)

E2_OUT = np.array([
    [2, 2, 2],
], dtype=int)

E3_IN = np.array([
    [0, 7, 0],
    [0, 0, 0],
    [0, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [7],
], dtype=int)

E4_IN = np.array([
    [0, 8, 0],
    [8, 8, 0],
    [8, 0, 0],
], dtype=int)

E4_OUT = np.array([
    [8, 8, 8, 8],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [4, 4, 0],
    [4, 0, 4],
    [0, 0, 4],
], dtype=int)

T_OUT = np.array([
    [4, 4, 4, 4, 4],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
    return [[x for x in sum(j, []) if x]]


# --- Code Golf Solution (Compressed) ---
def q(m):
    return [[*filter(int, sum(m, []))]]


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

def generate_d631b094(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    bgc = 0
    remcols = remove(bgc, cols)
    fgc = choice(remcols)
    nc = unifint(diff_lb, diff_ub, (1, min(30, (h * w) // 2 - 1)))
    c = canvas(bgc, (h, w))
    cands = totuple(asindices(c))
    cels = sample(cands, nc)
    gi = fill(c, fgc, cels)
    go = canvas(fgc, (1, nc))
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
IntegerSet = FrozenSet[Integer]

Element = Union[Object, Grid]

ZERO = 0

ONE = 1

def first(
    container: Container
) -> Any:
    """ first item of container """
    return next(iter(container))

def other(
    container: Container,
    value: Any
) -> Any:
    """ other value in the container """
    return first(remove(value, container))

def astuple(
    a: Integer,
    b: Integer
) -> IntegerTuple:
    """ constructs a tuple """
    return (a, b)

def colorcount(
    element: Element,
    value: Integer
) -> Integer:
    """ number of cells with color """
    if isinstance(element, tuple):
        return sum(row.count(value) for row in element)
    return sum(v == value for v, _ in element)

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

def verify_d631b094(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = palette(I)
    x1 = other(x0, ZERO)
    x2 = colorcount(I, x1)
    x3 = astuple(ONE, x2)
    x4 = canvas(x1, x3)
    return x4


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("E4", E4_IN, E4_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_d631b094(inp)
        assert pred == _to_grid(expected), f"{name} failed"

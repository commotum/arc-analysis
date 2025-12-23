# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "d4469b4b"
SERIAL = "334"
URL    = "https://arcprize.org/play?task=d4469b4b"

# --- Code Golf Concepts ---
CONCEPTS = [
    "dominant_color",
    "associate_images_to_colors",
]

# --- Example Grids ---
E1_IN = np.array([
    [2, 0, 0, 0, 0],
    [0, 2, 0, 0, 2],
    [2, 0, 0, 2, 0],
    [0, 0, 0, 2, 2],
    [0, 0, 2, 2, 0],
], dtype=int)

E1_OUT = np.array([
    [5, 5, 5],
    [0, 5, 0],
    [0, 5, 0],
], dtype=int)

E2_IN = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1],
    [0, 1, 0, 1, 1],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 0, 1],
], dtype=int)

E2_OUT = np.array([
    [0, 5, 0],
    [5, 5, 5],
    [0, 5, 0],
], dtype=int)

E3_IN = np.array([
    [3, 0, 0, 0, 0],
    [0, 0, 0, 3, 3],
    [0, 3, 3, 0, 0],
    [0, 3, 0, 3, 0],
    [3, 0, 3, 3, 0],
], dtype=int)

E3_OUT = np.array([
    [0, 0, 5],
    [0, 0, 5],
    [5, 5, 5],
], dtype=int)

E4_IN = np.array([
    [1, 0, 1, 0, 0],
    [1, 0, 0, 1, 1],
    [1, 1, 0, 1, 0],
    [0, 1, 0, 1, 0],
    [1, 0, 0, 0, 1],
], dtype=int)

E4_OUT = np.array([
    [0, 5, 0],
    [5, 5, 5],
    [0, 5, 0],
], dtype=int)

E5_IN = np.array([
    [2, 0, 2, 0, 2],
    [2, 0, 0, 0, 2],
    [2, 2, 0, 0, 0],
    [2, 0, 0, 2, 2],
    [2, 2, 2, 0, 2],
], dtype=int)

E5_OUT = np.array([
    [5, 5, 5],
    [0, 5, 0],
    [0, 5, 0],
], dtype=int)

E6_IN = np.array([
    [0, 2, 0, 2, 0],
    [0, 2, 2, 2, 0],
    [0, 2, 2, 0, 2],
    [2, 2, 2, 0, 0],
    [0, 0, 2, 0, 2],
], dtype=int)

E6_OUT = np.array([
    [5, 5, 5],
    [0, 5, 0],
    [0, 5, 0],
], dtype=int)

E7_IN = np.array([
    [0, 3, 0, 3, 0],
    [3, 3, 0, 0, 0],
    [0, 3, 0, 0, 0],
    [0, 0, 3, 0, 0],
    [3, 3, 3, 0, 0],
], dtype=int)

E7_OUT = np.array([
    [0, 0, 5],
    [0, 0, 5],
    [5, 5, 5],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [1, 1, 1, 1, 0],
    [0, 0, 1, 0, 1],
    [0, 1, 0, 0, 0],
    [0, 1, 0, 0, 1],
    [0, 0, 1, 0, 0],
], dtype=int)

T_OUT = np.array([
    [0, 5, 0],
    [5, 5, 5],
    [0, 5, 0],
], dtype=int)

T2_IN = np.array([
    [0, 3, 0, 3, 3],
    [0, 0, 3, 0, 0],
    [3, 0, 0, 0, 0],
    [0, 0, 3, 0, 3],
    [0, 0, 0, 0, 3],
], dtype=int)

T2_OUT = np.array([
    [0, 0, 5],
    [0, 0, 5],
    [5, 5, 5],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):A={2:[[5,5,5],[0,5,0],[0,5,0]],1:[[0,5,0],[5,5,5],[0,5,0]],3:[[0,0,5],[0,0,5],[5,5,5]]};c=[i for s in j for i in s];return A[max(c)]


# --- Code Golf Solution (Compressed) ---
def q(g):
    return [[i % 7, i % 6, i % 11] for i in b' M\x05~\x05M~MM\x05'[max(max(g))::3]]


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

def ulcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of upper left corner """
    return tuple(map(min, zip(*toindices(patch))))

def urcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of upper right corner """
    return tuple(map(lambda ix: {0: min, 1: max}[ix[0]](ix[1]), enumerate(zip(*toindices(patch)))))

def llcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of lower left corner """
    return tuple(map(lambda ix: {0: max, 1: min}[ix[0]](ix[1]), enumerate(zip(*toindices(patch)))))

def lrcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of lower right corner """
    return tuple(map(max, zip(*toindices(patch))))

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

def paint(
    grid: Grid,
    obj: Object
) -> Grid:
    """ paint object to grid """
    h, w = len(grid), len(grid[0])
    grid_painted = list(list(row) for row in grid)
    for value, (i, j) in obj:
        if 0 <= i < h and 0 <= j < w:
            grid_painted[i][j] = value
    return tuple(tuple(row) for row in grid_painted)

def canvas(
    value: Integer,
    dimensions: IntegerTuple
) -> Grid:
    """ grid construction """
    return tuple(tuple(value for j in range(dimensions[1])) for i in range(dimensions[0]))

def corners(
    patch: Patch
) -> Indices:
    """ indices of corners """
    return frozenset({ulcorner(patch), urcorner(patch), llcorner(patch), lrcorner(patch)})

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

def generate_d4469b4b(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2, 3))
    canv = canvas(5, (3, 3))
    A = fill(canv, 0, {(1, 0), (2, 0), (1, 2), (2, 2)})
    B = fill(canv, 0, corners(asindices(canv)))
    C = fill(canv, 0, {(0, 0), (0, 1), (1, 0), (1, 1)})
    colabc = ((2, A), (1, B), (3, C))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    col, go = choice(colabc)
    gi = canvas(col, (h, w))
    inds = asindices(gi)
    numc = unifint(diff_lb, diff_ub, (1, 7))
    ccols = sample(cols, numc)
    numcells = unifint(diff_lb, diff_ub, (0, h * w - 1))
    locs = sample(totuple(inds), numcells)
    otherobj = {(choice(ccols), ij) for ij in locs}
    gi = paint(gi, otherobj)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

IntegerSet = FrozenSet[Integer]

Element = Union[Object, Grid]

ZERO = 0

ONE = 1

TWO = 2

FIVE = 5

UNITY = (1, 1)

RIGHT = (0, 1)

TWO_BY_TWO = (2, 2)

THREE_BY_THREE = (3, 3)

def contained(
    value: Any,
    container: Container
) -> Boolean:
    """ element of """
    return value in container

def combine(
    a: Container,
    b: Container
) -> Container:
    """ union """
    return type(a)((*a, *b))

def branch(
    condition: Boolean,
    if_value: Any,
    else_value: Any
) -> Any:
    """ if else branching """
    return if_value if condition else else_value

def fork(
    outer: Callable,
    a: Callable,
    b: Callable
) -> Callable:
    """ creates a wrapper function """
    return lambda x: outer(a(x), b(x))

def palette(
    element: Element
) -> IntegerSet:
    """ colors occurring in object or grid """
    if isinstance(element, tuple):
        return frozenset({v for r in element for v in r})
    return frozenset({v for v, _ in element})

def vfrontier(
    location: IntegerTuple
) -> Indices:
    """ vertical frontier """
    return frozenset((i, location[1]) for i in range(30))

def hfrontier(
    location: IntegerTuple
) -> Indices:
    """ horizontal frontier """
    return frozenset((location[0], j) for j in range(30))

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_d4469b4b(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = palette(I)
    x1 = contained(ONE, x0)
    x2 = contained(TWO, x0)
    x3 = branch(x1, UNITY, TWO_BY_TWO)
    x4 = branch(x2, RIGHT, x3)
    x5 = fork(combine, vfrontier, hfrontier)
    x6 = x5(x4)
    x7 = canvas(ZERO, THREE_BY_THREE)
    x8 = fill(x7, FIVE, x6)
    return x8


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("E4", E4_IN, E4_OUT),
        ("E5", E5_IN, E5_OUT),
        ("E6", E6_IN, E6_OUT),
        ("E7", E7_IN, E7_OUT),
        ("T", T_IN, T_OUT),
        ("T2", T2_IN, T2_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_d4469b4b(inp)
        assert pred == _to_grid(expected), f"{name} failed"

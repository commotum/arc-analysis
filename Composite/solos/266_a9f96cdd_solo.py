# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "a9f96cdd"
SERIAL = "266"
URL    = "https://arcprize.org/play?task=a9f96cdd"

# --- Code Golf Concepts ---
CONCEPTS = [
    "replace_pattern",
    "out_of_boundary",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0],
    [0, 0, 0, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [3, 0, 6, 0, 0],
    [0, 0, 0, 0, 0],
    [8, 0, 7, 0, 0],
], dtype=int)

E2_IN = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 2],
], dtype=int)

E2_OUT = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 3, 0],
    [0, 0, 0, 0, 0],
], dtype=int)

E3_IN = np.array([
    [0, 0, 2, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [0, 0, 0, 0, 0],
    [0, 8, 0, 7, 0],
    [0, 0, 0, 0, 0],
], dtype=int)

E4_IN = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 2, 0],
    [0, 0, 0, 0, 0],
], dtype=int)

E4_OUT = np.array([
    [0, 0, 3, 0, 6],
    [0, 0, 0, 0, 0],
    [0, 0, 8, 0, 7],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 2],
    [0, 0, 0, 0, 0],
], dtype=int)

T_OUT = np.array([
    [0, 0, 0, 3, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 8, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
 A=sum(j,[]).index(2);c,E=divmod(A,5);j[c][E]=0
 if c*E:j[c-1][E-1]=3
 if c<2 and E:j[c+1][E-1]=8
 if E<4 and c:j[c-1][E+1]=6
 if c<2 and E<4:j[c+1][E+1]=7
 return j


# --- Code Golf Solution (Compressed) ---
def q(g):
    return g[9:] or [[(8 | 9 >> c | a * 6) % 9 for a, c in zip([0] + r, r[1:] + [0])] for *r, in zip(*p(g * 2))]


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, uniform

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

ContainerContainer = Container[Container]

def difference(
    a: Container,
    b: Container
) -> Container:
    """ difference """
    return type(a)(e for e in a if e not in b)

def merge(
    containers: ContainerContainer
) -> Container:
    """ merging """
    return type(containers)(e for c in containers for e in c)

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

def apply(
    function: Callable,
    container: Container
) -> Container:
    """ apply function to each item in container """
    return type(container)(function(e) for e in container)

def mapply(
    function: Callable,
    container: ContainerContainer
) -> FrozenSet:
    """ apply and merge """
    return merge(apply(function, container))

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

def shift(
    patch: Patch,
    directions: IntegerTuple
) -> Patch:
    """ shift patch """
    if len(patch) == 0:
        return patch
    di, dj = directions
    if isinstance(next(iter(patch))[1], tuple):
        return frozenset((value, (i + di, j + dj)) for value, (i, j) in patch)
    return frozenset((i + di, j + dj) for i, j in patch)

def dneighbors(
    loc: IntegerTuple
) -> Indices:
    """ directly adjacent indices """
    return frozenset({(loc[0] - 1, loc[1]), (loc[0] + 1, loc[1]), (loc[0], loc[1] - 1), (loc[0], loc[1] + 1)})

def ineighbors(
    loc: IntegerTuple
) -> Indices:
    """ diagonally adjacent indices """
    return frozenset({(loc[0] - 1, loc[1] - 1), (loc[0] - 1, loc[1] + 1), (loc[0] + 1, loc[1] - 1), (loc[0] + 1, loc[1] + 1)})

def neighbors(
    loc: IntegerTuple
) -> Indices:
    """ adjacent indices """
    return dneighbors(loc) | ineighbors(loc)

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

def generate_a9f96cdd(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (3, 6, 7, 8))
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc = choice(cols)
    fgc = choice(remove(bgc, cols))
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    locs = asindices(gi)
    noccs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 10)))
    for k in range(noccs):
        if len(locs) == 0:
            break
        loc = choice(totuple(locs))
        locs = locs - mapply(neighbors, neighbors(loc))
        plcd = {loc}
        gi = fill(gi, fgc, plcd)
        go = fill(go, 3, shift(plcd, (-1, -1)))
        go = fill(go, 7, shift(plcd, (1, 1)))
        go = fill(go, 8, shift(plcd, (1, -1)))
        go = fill(go, 6, shift(plcd, (-1, 1)))
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Element = Union[Object, Grid]

THREE = 3

SIX = 6

SEVEN = 7

EIGHT = 8

UNITY = (1, 1)

NEG_UNITY = (-1, -1)

UP_RIGHT = (-1, 1)

DOWN_LEFT = (1, -1)

def combine(
    a: Container,
    b: Container
) -> Container:
    """ union """
    return type(a)((*a, *b))

def mostcolor(
    element: Element
) -> Integer:
    """ most common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return max(set(values), key=values.count)

def leastcolor(
    element: Element
) -> Integer:
    """ least common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return min(set(values), key=values.count)

def ofcolor(
    grid: Grid,
    value: Integer
) -> Indices:
    """ indices of all grid cells with value """
    return frozenset((i, j) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value)

def recolor(
    value: Integer,
    patch: Patch
) -> Object:
    """ recolor patch """
    return frozenset((value, index) for index in toindices(patch))

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

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_a9f96cdd(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = leastcolor(I)
    x1 = ofcolor(I, x0)
    x2 = shift(x1, NEG_UNITY)
    x3 = recolor(THREE, x2)
    x4 = shift(x1, UNITY)
    x5 = recolor(SEVEN, x4)
    x6 = shift(x1, DOWN_LEFT)
    x7 = recolor(EIGHT, x6)
    x8 = shift(x1, UP_RIGHT)
    x9 = recolor(SIX, x8)
    x10 = mostcolor(I)
    x11 = fill(I, x10, x1)
    x12 = combine(x3, x5)
    x13 = combine(x7, x9)
    x14 = combine(x12, x13)
    x15 = paint(x11, x14)
    return x15


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("E4", E4_IN, E4_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_a9f96cdd(inp)
        assert pred == _to_grid(expected), f"{name} failed"

# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "f5b8619d"
SERIAL = "388"
URL    = "https://arcprize.org/play?task=f5b8619d"

# --- Code Golf Concepts ---
CONCEPTS = [
    "pattern_expansion",
    "draw_line_from_point",
    "image_repetition",
]

# --- Example Grids ---
E1_IN = np.array([
    [2, 0, 0],
    [0, 0, 0],
    [0, 0, 2],
], dtype=int)

E1_OUT = np.array([
    [2, 0, 8, 2, 0, 8],
    [8, 0, 8, 8, 0, 8],
    [8, 0, 2, 8, 0, 2],
    [2, 0, 8, 2, 0, 8],
    [8, 0, 8, 8, 0, 8],
    [8, 0, 2, 8, 0, 2],
], dtype=int)

E2_IN = np.array([
    [0, 5, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [5, 0, 0, 0, 0, 5],
    [0, 0, 0, 0, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [8, 5, 0, 0, 0, 8, 8, 5, 0, 0, 0, 8],
    [8, 8, 0, 0, 0, 8, 8, 8, 0, 0, 0, 8],
    [8, 8, 0, 0, 0, 8, 8, 8, 0, 0, 0, 8],
    [8, 8, 0, 0, 0, 8, 8, 8, 0, 0, 0, 8],
    [5, 8, 0, 0, 0, 5, 5, 8, 0, 0, 0, 5],
    [8, 8, 0, 0, 0, 8, 8, 8, 0, 0, 0, 8],
    [8, 5, 0, 0, 0, 8, 8, 5, 0, 0, 0, 8],
    [8, 8, 0, 0, 0, 8, 8, 8, 0, 0, 0, 8],
    [8, 8, 0, 0, 0, 8, 8, 8, 0, 0, 0, 8],
    [8, 8, 0, 0, 0, 8, 8, 8, 0, 0, 0, 8],
    [5, 8, 0, 0, 0, 5, 5, 8, 0, 0, 0, 5],
    [8, 8, 0, 0, 0, 8, 8, 8, 0, 0, 0, 8],
], dtype=int)

E3_IN = np.array([
    [0, 4],
    [0, 0],
], dtype=int)

E3_OUT = np.array([
    [0, 4, 0, 4],
    [0, 8, 0, 8],
    [0, 4, 0, 4],
    [0, 8, 0, 8],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 3, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 3],
    [3, 0, 0, 0],
], dtype=int)

T_OUT = np.array([
    [8, 0, 3, 8, 8, 0, 3, 8],
    [8, 0, 8, 8, 8, 0, 8, 8],
    [8, 0, 8, 3, 8, 0, 8, 3],
    [3, 0, 8, 8, 3, 0, 8, 8],
    [8, 0, 3, 8, 8, 0, 3, 8],
    [8, 0, 8, 8, 8, 0, 8, 8],
    [8, 0, 8, 3, 8, 0, 8, 3],
    [3, 0, 8, 8, 3, 0, 8, 8],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(g):R=range;n=len(g);c={j for i in R(n)for j in R(n)if g[i][j]};m=[[8if g[i][j]==0 and j in c else g[i][j]for j in R(n)]for i in R(n)];return[[m[i%n][j%n]for j in R(2*n)]for i in R(2*n)]


# --- Code Golf Solution (Compressed) ---
def q(g):
    return [[x | 8 & x - any(c) for *c, x in zip(*g, r)] * 2 for r in g] * 2


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

ContainerContainer = Container[Container]

def merge(
    containers: ContainerContainer
) -> Container:
    """ merging """
    return type(containers)(e for c in containers for e in c)

def sfilter(
    container: Container,
    condition: Callable
) -> Container:
    """ keep elements in container that satisfy condition """
    return type(container)(e for e in container if condition(e))

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

def hconcat(
    a: Grid,
    b: Grid
) -> Grid:
    """ concatenate two grids horizontally """
    return tuple(i + j for i, j in zip(a, b))

def vconcat(
    a: Grid,
    b: Grid
) -> Grid:
    """ concatenate two grids vertically """
    return a + b

def canvas(
    value: Integer,
    dimensions: IntegerTuple
) -> Grid:
    """ grid construction """
    return tuple(tuple(value for j in range(dimensions[1])) for i in range(dimensions[0]))

def vfrontier(
    location: IntegerTuple
) -> Indices:
    """ vertical frontier """
    return frozenset((i, location[1]) for i in range(30))

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

def generate_f5b8619d(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(8, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 15))
    w = unifint(diff_lb, diff_ub, (2, 15))
    ncells = unifint(diff_lb, diff_ub, (1, (h * w) // 2 - 1))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    inds = asindices(gi)
    locs = sample(totuple(inds), ncells)
    blockcol = randint(0, w - 1)
    locs = sfilter(locs, lambda ij: ij[1] != blockcol)
    numcols = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, numcols)
    obj = frozenset({(choice(ccols), ij) for ij in locs})
    gi = paint(gi, obj)
    go = fill(gi, 8, mapply(vfrontier, set(locs)) & (inds - set(locs)))
    go = hconcat(go, go)
    go = vconcat(go, go)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
IntegerSet = FrozenSet[Integer]

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

EIGHT = 8

def mostcolor(
    element: Element
) -> Integer:
    """ most common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return max(set(values), key=values.count)

def fgpartition(
    grid: Grid
) -> Objects:
    """ each cell with the same value part of the same object without background """
    return frozenset(
        frozenset(
            (v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value
        ) for value in palette(grid) - {mostcolor(grid)}
    )

def palette(
    element: Element
) -> IntegerSet:
    """ colors occurring in object or grid """
    if isinstance(element, tuple):
        return frozenset({v for r in element for v in r})
    return frozenset({v for v, _ in element})

def underfill(
    grid: Grid,
    value: Integer,
    patch: Patch
) -> Grid:
    """ fill value at indices that are background """
    h, w = len(grid), len(grid[0])
    bg = mostcolor(grid)
    grid_filled = list(list(row) for row in grid)
    for i, j in toindices(patch):
        if 0 <= i < h and 0 <= j < w:
            if grid_filled[i][j] == bg:
                grid_filled[i][j] = value
    return tuple(tuple(row) for row in grid_filled)

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_f5b8619d(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = fgpartition(I)
    x1 = mapply(toindices, x0)
    x2 = mapply(vfrontier, x1)
    x3 = underfill(I, EIGHT, x2)
    x4 = hconcat(x3, x3)
    x5 = vconcat(x4, x4)
    return x5


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_f5b8619d(inp)
        assert pred == _to_grid(expected), f"{name} failed"

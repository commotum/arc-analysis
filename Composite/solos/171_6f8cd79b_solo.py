# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "6f8cd79b"
SERIAL = "171"
URL    = "https://arcprize.org/play?task=6f8cd79b"

# --- Code Golf Concepts ---
CONCEPTS = [
    "ex_nihilo",
    "contouring",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [8, 8, 8],
    [8, 0, 8],
    [8, 8, 8],
], dtype=int)

E2_IN = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [8, 8, 8],
    [8, 0, 8],
    [8, 0, 8],
    [8, 8, 8],
], dtype=int)

E3_IN = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [8, 8, 8, 8],
    [8, 0, 0, 8],
    [8, 0, 0, 8],
    [8, 0, 0, 8],
    [8, 8, 8, 8],
], dtype=int)

E4_IN = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
], dtype=int)

E4_OUT = np.array([
    [8, 8, 8, 8, 8, 8],
    [8, 0, 0, 0, 0, 8],
    [8, 0, 0, 0, 0, 8],
    [8, 0, 0, 0, 0, 8],
    [8, 8, 8, 8, 8, 8],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
], dtype=int)

T_OUT = np.array([
    [8, 8, 8, 8, 8, 8],
    [8, 0, 0, 0, 0, 8],
    [8, 0, 0, 0, 0, 8],
    [8, 0, 0, 0, 0, 8],
    [8, 0, 0, 0, 0, 8],
    [8, 0, 0, 0, 0, 8],
    [8, 8, 8, 8, 8, 8],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(g):
 g[-1]=g[0]=[8]*len(g[0])
 for r in range(len(g)):g[r][0]=8;g[r][-1]=8
 return g


# --- Code Golf Solution (Compressed) ---
def q(*m):
    return [*zip((a := ([8] * 9)), *(m[2:] or p(*m[0])[2:]), a)]


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

def ulcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of upper left corner """
    return tuple(map(min, zip(*toindices(patch))))

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

def box(
    patch: Patch
) -> Indices:
    """ outline of patch """
    if len(patch) == 0:
        return patch
    ai, aj = ulcorner(patch)
    bi, bj = lrcorner(patch)
    si, sj = min(ai, bi), min(aj, bj)
    ei, ej = max(ai, bi), max(aj, bj)
    vlines = {(i, sj) for i in range(si, ei + 1)} | {(i, ej) for i in range(si, ei + 1)}
    hlines = {(si, j) for j in range(sj, ej + 1)} | {(ei, j) for j in range(sj, ej + 1)}
    return frozenset(vlines | hlines)

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

def generate_6f8cd79b(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(8, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    ncols = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, ncols)
    ncells = unifint(diff_lb, diff_ub, (0, h * w))
    inds = asindices(gi)
    cells = sample(totuple(inds), ncells)
    obj = {(choice(ccols), ij) for ij in cells}
    gi = paint(gi, obj)
    brd = box(inds)
    go = fill(gi, 8, brd)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
EIGHT = 8

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_6f8cd79b(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = asindices(I)
    x1 = box(x0)
    x2 = fill(I, EIGHT, x1)
    return x2


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("E4", E4_IN, E4_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_6f8cd79b(inp)
        assert pred == _to_grid(expected), f"{name} failed"

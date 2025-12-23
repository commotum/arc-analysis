# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "67e8384a"
SERIAL = "152"
URL    = "https://arcprize.org/play?task=67e8384a"

# --- Code Golf Concepts ---
CONCEPTS = [
    "image_repetition",
    "image_reflection",
    "image_rotation",
]

# --- Example Grids ---
E1_IN = np.array([
    [5, 3, 4],
    [3, 4, 5],
    [3, 4, 4],
], dtype=int)

E1_OUT = np.array([
    [5, 3, 4, 4, 3, 5],
    [3, 4, 5, 5, 4, 3],
    [3, 4, 4, 4, 4, 3],
    [3, 4, 4, 4, 4, 3],
    [3, 4, 5, 5, 4, 3],
    [5, 3, 4, 4, 3, 5],
], dtype=int)

E2_IN = np.array([
    [7, 1, 5],
    [7, 7, 1],
    [5, 3, 1],
], dtype=int)

E2_OUT = np.array([
    [7, 1, 5, 5, 1, 7],
    [7, 7, 1, 1, 7, 7],
    [5, 3, 1, 1, 3, 5],
    [5, 3, 1, 1, 3, 5],
    [7, 7, 1, 1, 7, 7],
    [7, 1, 5, 5, 1, 7],
], dtype=int)

E3_IN = np.array([
    [2, 5, 2],
    [2, 6, 4],
    [2, 2, 2],
], dtype=int)

E3_OUT = np.array([
    [2, 5, 2, 2, 5, 2],
    [2, 6, 4, 4, 6, 2],
    [2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2],
    [2, 6, 4, 4, 6, 2],
    [2, 5, 2, 2, 5, 2],
], dtype=int)

E4_IN = np.array([
    [1, 2, 1],
    [2, 8, 1],
    [8, 1, 6],
], dtype=int)

E4_OUT = np.array([
    [1, 2, 1, 1, 2, 1],
    [2, 8, 1, 1, 8, 2],
    [8, 1, 6, 6, 1, 8],
    [8, 1, 6, 6, 1, 8],
    [2, 8, 1, 1, 8, 2],
    [1, 2, 1, 1, 2, 1],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [1, 6, 6],
    [5, 2, 2],
    [2, 2, 2],
], dtype=int)

T_OUT = np.array([
    [1, 6, 6, 6, 6, 1],
    [5, 2, 2, 2, 2, 5],
    [2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2],
    [5, 2, 2, 2, 2, 5],
    [1, 6, 6, 6, 6, 1],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):A=[r+r[::-1]for r in j];return A+A[::-1]


# --- Code Golf Solution (Compressed) ---
def q(g):
    return g * -1 * -1 or [*map(p, g + g[::-1])]


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

Piece = Union[Grid, Patch]

ContainerContainer = Container[Container]

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

def hmirror(
    piece: Piece
) -> Piece:
    """ mirroring along horizontal """
    if isinstance(piece, tuple):
        return piece[::-1]
    d = ulcorner(piece)[0] + lrcorner(piece)[0]
    if isinstance(next(iter(piece))[1], tuple):
        return frozenset((v, (d - i, j)) for v, (i, j) in piece)
    return frozenset((d - i, j) for i, j in piece)

def vmirror(
    piece: Piece
) -> Piece:
    """ mirroring along vertical """
    if isinstance(piece, tuple):
        return tuple(row[::-1] for row in piece)
    d = ulcorner(piece)[1] + lrcorner(piece)[1]
    if isinstance(next(iter(piece))[1], tuple):
        return frozenset((v, (i, d - j)) for v, (i, j) in piece)
    return frozenset((i, d - j) for i, j in piece)

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

def generate_67e8384a(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 14))
    w = unifint(diff_lb, diff_ub, (1, 14))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, 9))
    remcols = sample(remcols, numcols)
    canv = canvas(bgc, (h, w))
    nc = unifint(diff_lb, diff_ub, (1, h * w))
    bx = asindices(canv)
    obj = {(choice(remcols), choice(totuple(bx)))}
    for kk in range(nc - 1):
        dns = mapply(neighbors, toindices(obj))
        ch = choice(totuple(bx & dns))
        obj.add((choice(remcols), ch))
        bx = bx - {ch}
    gi = paint(canv, obj)
    go = paint(canv, obj)
    go = hconcat(go, vmirror(go))
    go = vconcat(go, hmirror(go))
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_67e8384a(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = vmirror(I)
    x1 = hconcat(I, x0)
    x2 = hmirror(x1)
    x3 = vconcat(x1, x2)
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
        pred = verify_67e8384a(inp)
        assert pred == _to_grid(expected), f"{name} failed"

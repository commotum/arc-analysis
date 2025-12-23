# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "49d1d64f"
SERIAL = "114"
URL    = "https://arcprize.org/play?task=49d1d64f"

# --- Code Golf Concepts ---
CONCEPTS = [
    "pattern_expansion",
    "image_expansion",
]

# --- Example Grids ---
E1_IN = np.array([
    [1, 2],
    [3, 8],
], dtype=int)

E1_OUT = np.array([
    [0, 1, 2, 0],
    [1, 1, 2, 2],
    [3, 3, 8, 8],
    [0, 3, 8, 0],
], dtype=int)

E2_IN = np.array([
    [1, 8, 4],
    [8, 3, 8],
], dtype=int)

E2_OUT = np.array([
    [0, 1, 8, 4, 0],
    [1, 1, 8, 4, 4],
    [8, 8, 3, 8, 8],
    [0, 8, 3, 8, 0],
], dtype=int)

E3_IN = np.array([
    [2, 1, 4],
    [8, 0, 2],
    [3, 2, 8],
], dtype=int)

E3_OUT = np.array([
    [0, 2, 1, 4, 0],
    [2, 2, 1, 4, 4],
    [8, 8, 0, 2, 2],
    [3, 3, 2, 8, 8],
    [0, 3, 2, 8, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [2, 8],
    [1, 4],
    [3, 4],
], dtype=int)

T_OUT = np.array([
    [0, 2, 8, 0],
    [2, 2, 8, 8],
    [1, 1, 4, 4],
    [3, 3, 4, 4],
    [0, 3, 4, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(g):
 g=[g[0]]+g+[g[-1]]
 g=[[R[0]]+R+[R[-1]]for R in g]
 for r,c in[[0,0],[0,-1],[-1,0],[-1,-1]]:g[r][c]=0
 return g


# --- Code Golf Solution (Compressed) ---
def q(g, v=1):
    return g * 0 != 0 and [v * p(g[0], 0), *map(p, g), v * p(g[-1], 0)] or g


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

def sfilter(
    container: Container,
    condition: Callable
) -> Container:
    """ keep elements in container that satisfy condition """
    return type(container)(e for e in container if condition(e))

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

def asobject(
    grid: Grid
) -> Object:
    """ conversion of grid to object """
    return frozenset((v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r))

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

def generate_49d1d64f(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 28))
    w = unifint(diff_lb, diff_ub, (2, 28))
    ncols = unifint(diff_lb, diff_ub, (1, 10))
    ccols = sample(cols, ncols)
    gi = canvas(-1, (h, w))
    obj = {(choice(ccols), ij) for ij in asindices(gi)}
    gi = paint(gi, obj)
    go = canvas(0, (h+2, w+2))
    go = paint(go, shift(asobject(gi), (1, 1)))
    ts = sfilter(obj, lambda cij: cij[1][0] == 0)
    bs = sfilter(obj, lambda cij: cij[1][0] == h - 1)
    ls = sfilter(obj, lambda cij: cij[1][1] == 0)
    rs = sfilter(obj, lambda cij: cij[1][1] == w - 1)
    ts = shift(ts, (1, 1))
    bs = shift(bs, (1, 1))
    ls = shift(ls, (1, 1))
    rs = shift(rs, (1, 1))
    go = paint(go, shift(ts, (-1, 0)))
    go = paint(go, shift(bs, (1, 0)))
    go = paint(go, shift(ls, (0, -1)))
    go = paint(go, shift(rs, (0, 1)))
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Numerical = Union[Integer, IntegerTuple]

Piece = Union[Grid, Patch]

ZERO = 0

UNITY = (1, 1)

DOWN = (1, 0)

RIGHT = (0, 1)

UP = (-1, 0)

LEFT = (0, -1)

def increment(
    x: Numerical
) -> Numerical:
    """ incrementing """
    return x + 1 if isinstance(x, int) else (x[0] + 1, x[1] + 1)

def height(
    piece: Piece
) -> Integer:
    """ height of grid or patch """
    if len(piece) == 0:
        return 0
    if isinstance(piece, tuple):
        return len(piece)
    return lowermost(piece) - uppermost(piece) + 1

def width(
    piece: Piece
) -> Integer:
    """ width of grid or patch """
    if len(piece) == 0:
        return 0
    if isinstance(piece, tuple):
        return len(piece[0])
    return rightmost(piece) - leftmost(piece) + 1

def shape(
    piece: Piece
) -> IntegerTuple:
    """ height and width of grid or patch """
    return (height(piece), width(piece))

def toindices(
    patch: Patch
) -> Indices:
    """ indices of object cells """
    if len(patch) == 0:
        return frozenset()
    if isinstance(next(iter(patch))[1], tuple):
        return frozenset(index for value, index in patch)
    return patch

def uppermost(
    patch: Patch
) -> Integer:
    """ row index of uppermost occupied cell """
    return min(i for i, j in toindices(patch))

def lowermost(
    patch: Patch
) -> Integer:
    """ row index of lowermost occupied cell """
    return max(i for i, j in toindices(patch))

def leftmost(
    patch: Patch
) -> Integer:
    """ column index of leftmost occupied cell """
    return min(j for i, j in toindices(patch))

def rightmost(
    patch: Patch
) -> Integer:
    """ column index of rightmost occupied cell """
    return max(j for i, j in toindices(patch))

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_49d1d64f(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = shape(I)
    x1 = increment(x0)
    x2 = increment(x1)
    x3 = canvas(ZERO, x2)
    x4 = asobject(I)
    x5 = shift(x4, UNITY)
    x6 = shift(x5, LEFT)
    x7 = paint(x3, x6)
    x8 = shift(x5, RIGHT)
    x9 = paint(x7, x8)
    x10 = shift(x5, UP)
    x11 = paint(x9, x10)
    x12 = shift(x5, DOWN)
    x13 = paint(x11, x12)
    x14 = paint(x13, x5)
    return x14


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_49d1d64f(inp)
        assert pred == _to_grid(expected), f"{name} failed"

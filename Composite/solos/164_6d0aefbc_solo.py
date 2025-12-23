# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "6d0aefbc"
SERIAL = "164"
URL    = "https://arcprize.org/play?task=6d0aefbc"

# --- Code Golf Concepts ---
CONCEPTS = [
    "image_repetition",
    "image_reflection",
]

# --- Example Grids ---
E1_IN = np.array([
    [6, 6, 6],
    [1, 6, 1],
    [8, 8, 6],
], dtype=int)

E1_OUT = np.array([
    [6, 6, 6, 6, 6, 6],
    [1, 6, 1, 1, 6, 1],
    [8, 8, 6, 6, 8, 8],
], dtype=int)

E2_IN = np.array([
    [6, 8, 1],
    [6, 1, 1],
    [1, 1, 6],
], dtype=int)

E2_OUT = np.array([
    [6, 8, 1, 1, 8, 6],
    [6, 1, 1, 1, 1, 6],
    [1, 1, 6, 6, 1, 1],
], dtype=int)

E3_IN = np.array([
    [1, 1, 1],
    [8, 1, 6],
    [6, 8, 8],
], dtype=int)

E3_OUT = np.array([
    [1, 1, 1, 1, 1, 1],
    [8, 1, 6, 6, 1, 8],
    [6, 8, 8, 8, 8, 6],
], dtype=int)

E4_IN = np.array([
    [1, 1, 1],
    [1, 6, 6],
    [6, 6, 6],
], dtype=int)

E4_OUT = np.array([
    [1, 1, 1, 1, 1, 1],
    [1, 6, 6, 6, 6, 1],
    [6, 6, 6, 6, 6, 6],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [6, 8, 6],
    [8, 6, 8],
    [1, 6, 1],
], dtype=int)

T_OUT = np.array([
    [6, 8, 6, 6, 8, 6],
    [8, 6, 8, 8, 6, 8],
    [1, 6, 1, 1, 6, 1],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
    return [R + R[::-1] for R in j]


# --- Code Golf Solution (Compressed) ---
def q(m):
    return [r + r[::-1] for r in m]


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

def hconcat(
    a: Grid,
    b: Grid
) -> Grid:
    """ concatenate two grids horizontally """
    return tuple(i + j for i, j in zip(a, b))

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

def generate_6d0aefbc(diff_lb: float, diff_ub: float) -> dict:
    dim_bounds = (1, 30)
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 30))
    w = unifint(diff_lb, diff_ub, (1, 15))
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = sample(remcols, numc)
    inds = totuple(asindices(gi))
    for col in colsch:
        num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = sample(inds, num)
        gi = fill(gi, col, chos)
        inds = difference(inds, chos)
    go = hconcat(gi, vmirror(gi))
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

def verify_6d0aefbc(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = vmirror(I)
    x1 = hconcat(I, x0)
    return x1


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("E4", E4_IN, E4_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_6d0aefbc(inp)
        assert pred == _to_grid(expected), f"{name} failed"

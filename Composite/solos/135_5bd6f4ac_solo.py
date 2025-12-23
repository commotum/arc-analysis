# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "5bd6f4ac"
SERIAL = "135"
URL    = "https://arcprize.org/play?task=5bd6f4ac"

# --- Code Golf Concepts ---
CONCEPTS = [
    "rectangle_guessing",
    "crop",
]

# --- Example Grids ---
E1_IN = np.array([
    [3, 0, 0, 7, 0, 0, 9, 7, 0],
    [8, 4, 0, 6, 6, 0, 4, 8, 4],
    [1, 7, 0, 0, 0, 0, 4, 0, 0],
    [1, 1, 0, 9, 1, 0, 7, 0, 0],
    [0, 0, 0, 0, 7, 7, 0, 0, 0],
    [8, 0, 0, 1, 7, 0, 8, 4, 0],
    [0, 7, 0, 9, 9, 2, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 5, 0, 0],
    [0, 0, 0, 2, 4, 0, 8, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [9, 7, 0],
    [4, 8, 4],
    [4, 0, 0],
], dtype=int)

E2_IN = np.array([
    [9, 0, 0, 0, 0, 0, 0, 6, 0],
    [0, 4, 0, 7, 0, 5, 0, 8, 1],
    [0, 2, 0, 0, 7, 1, 4, 4, 5],
    [0, 6, 0, 0, 4, 0, 0, 0, 0],
    [8, 3, 0, 4, 2, 0, 0, 9, 7],
    [0, 0, 2, 3, 0, 2, 0, 6, 7],
    [4, 0, 4, 0, 3, 4, 7, 0, 7],
    [7, 1, 0, 0, 0, 0, 3, 0, 0],
    [3, 2, 0, 0, 4, 0, 0, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [0, 6, 0],
    [0, 8, 1],
    [4, 4, 5],
], dtype=int)

E3_IN = np.array([
    [2, 5, 0, 0, 6, 0, 0, 0, 0],
    [2, 5, 5, 7, 0, 0, 6, 0, 1],
    [0, 3, 0, 0, 0, 1, 9, 4, 0],
    [0, 7, 0, 6, 0, 0, 0, 0, 0],
    [0, 9, 0, 0, 0, 1, 0, 0, 8],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 4, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 4],
    [0, 5, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [0, 0, 0],
    [6, 0, 1],
    [9, 4, 0],
], dtype=int)

E4_IN = np.array([
    [0, 5, 0, 0, 8, 0, 0, 0, 4],
    [0, 0, 0, 0, 0, 0, 3, 0, 0],
    [0, 0, 0, 0, 2, 1, 0, 0, 3],
    [0, 1, 0, 0, 0, 0, 3, 0, 0],
    [1, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 8, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 9, 4, 0, 0, 0, 0, 0],
    [3, 0, 7, 0, 0, 2, 0, 0, 6],
], dtype=int)

E4_OUT = np.array([
    [0, 0, 4],
    [3, 0, 0],
    [0, 0, 3],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [6, 9, 0, 0, 1, 0, 5, 8, 9],
    [2, 9, 0, 6, 0, 8, 0, 9, 0],
    [0, 0, 0, 0, 0, 9, 9, 2, 0],
    [9, 2, 6, 0, 0, 8, 0, 6, 8],
    [7, 7, 4, 0, 7, 0, 9, 0, 0],
    [0, 0, 7, 0, 0, 1, 5, 7, 4],
    [4, 1, 0, 0, 7, 5, 0, 0, 9],
    [9, 9, 0, 0, 0, 0, 1, 0, 0],
    [4, 9, 2, 0, 0, 0, 8, 4, 0],
], dtype=int)

T_OUT = np.array([
    [5, 8, 9],
    [0, 9, 0],
    [9, 2, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(g):
    return [r[6:9] for r in g[:3]]


# --- Code Golf Solution (Compressed) ---
def q(m):
    return [r[6:] for r in m[:3]]


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

def crop(
    grid: Grid,
    start: IntegerTuple,
    dims: IntegerTuple
) -> Grid:
    """ subgrid specified by start and dimension """
    return tuple(r[start[1]:start[1]+dims[1]] for r in grid[start[0]:start[0]+dims[0]])

def toindices(
    patch: Patch
) -> Indices:
    """ indices of object cells """
    if len(patch) == 0:
        return frozenset()
    if isinstance(next(iter(patch))[1], tuple):
        return frozenset(index for value, index in patch)
    return patch

def rot90(
    grid: Grid
) -> Grid:
    """ quarter clockwise rotation """
    return tuple(row for row in zip(*grid[::-1]))

def rot270(
    grid: Grid
) -> Grid:
    """ quarter anticlockwise rotation """
    return tuple(tuple(row[::-1]) for row in zip(*grid[::-1]))[::-1]

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

def generate_5bd6f4ac(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
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
    go = rot90(crop(rot270(gi), (0, 0), (3, 3)))
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
ORIGIN = (0, 0)

THREE_BY_THREE = (3, 3)

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_5bd6f4ac(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = rot270(I)
    x1 = crop(x0, ORIGIN, THREE_BY_THREE)
    x2 = rot90(x1)
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
        pred = verify_5bd6f4ac(inp)
        assert pred == _to_grid(expected), f"{name} failed"

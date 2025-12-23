# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "d10ecb37"
SERIAL = "326"
URL    = "https://arcprize.org/play?task=d10ecb37"

# --- Code Golf Concepts ---
CONCEPTS = [
    "crop",
]

# --- Example Grids ---
E1_IN = np.array([
    [4, 3, 6, 4, 0, 6],
    [6, 0, 0, 3, 3, 4],
    [6, 4, 4, 3, 3, 0],
    [0, 3, 6, 0, 4, 6],
    [0, 6, 3, 0, 4, 3],
    [3, 4, 4, 6, 6, 0],
], dtype=int)

E1_OUT = np.array([
    [4, 3],
    [6, 0],
], dtype=int)

E2_IN = np.array([
    [2, 4, 2, 2, 5, 2, 4, 5],
    [2, 5, 5, 4, 4, 2, 2, 2],
    [4, 5, 5, 2, 2, 2, 2, 4],
    [2, 2, 4, 2, 5, 4, 2, 5],
    [2, 4, 2, 2, 5, 2, 4, 5],
    [2, 5, 5, 4, 4, 2, 2, 2],
    [4, 5, 5, 2, 2, 2, 2, 4],
    [2, 2, 4, 2, 5, 4, 2, 5],
], dtype=int)

E2_OUT = np.array([
    [2, 4],
    [2, 5],
], dtype=int)

E3_IN = np.array([
    [3, 2, 1, 3, 4, 1],
    [1, 4, 4, 2, 2, 3],
    [1, 3, 3, 2, 2, 4],
    [4, 2, 1, 4, 3, 1],
    [4, 1, 2, 4, 3, 2],
    [2, 3, 3, 1, 1, 4],
    [2, 4, 4, 1, 1, 3],
    [3, 1, 2, 3, 4, 2],
    [3, 2, 1, 3, 4, 1],
    [1, 4, 4, 2, 2, 3],
    [1, 3, 3, 2, 2, 4],
    [4, 2, 1, 4, 3, 1],
], dtype=int)

E3_OUT = np.array([
    [3, 2],
    [1, 4],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [9, 6, 2, 9, 9, 2, 6, 9],
    [2, 9, 9, 6, 6, 9, 9, 2],
    [6, 9, 9, 2, 2, 9, 9, 6],
    [9, 2, 6, 9, 9, 6, 2, 9],
], dtype=int)

T_OUT = np.array([
    [9, 6],
    [2, 9],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
    return [r[:2] for r in j[:2]]


# --- Code Golf Solution (Compressed) ---
def q(m):
    return [m[0][:2], m[1][:2]]


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

def generate_d10ecb37(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
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
    go = crop(gi, (0, 0), (2, 2))
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
ORIGIN = (0, 0)

TWO_BY_TWO = (2, 2)

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_d10ecb37(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = crop(I, ORIGIN, TWO_BY_TWO)
    return x0


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_d10ecb37(inp)
        assert pred == _to_grid(expected), f"{name} failed"

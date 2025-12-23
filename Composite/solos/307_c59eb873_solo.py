# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "c59eb873"
SERIAL = "307"
URL    = "https://arcprize.org/play?task=c59eb873"

# --- Code Golf Concepts ---
CONCEPTS = [
    "image_resizing",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 5, 1],
    [5, 5, 5],
    [2, 5, 0],
], dtype=int)

E1_OUT = np.array([
    [0, 0, 5, 5, 1, 1],
    [0, 0, 5, 5, 1, 1],
    [5, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5, 5],
    [2, 2, 5, 5, 0, 0],
    [2, 2, 5, 5, 0, 0],
], dtype=int)

E2_IN = np.array([
    [2, 1],
    [3, 1],
], dtype=int)

E2_OUT = np.array([
    [2, 2, 1, 1],
    [2, 2, 1, 1],
    [3, 3, 1, 1],
    [3, 3, 1, 1],
], dtype=int)

E3_IN = np.array([
    [2, 0, 3, 0],
    [2, 1, 3, 0],
    [0, 0, 3, 3],
    [0, 0, 3, 5],
], dtype=int)

E3_OUT = np.array([
    [2, 2, 0, 0, 3, 3, 0, 0],
    [2, 2, 0, 0, 3, 3, 0, 0],
    [2, 2, 1, 1, 3, 3, 0, 0],
    [2, 2, 1, 1, 3, 3, 0, 0],
    [0, 0, 0, 0, 3, 3, 3, 3],
    [0, 0, 0, 0, 3, 3, 3, 3],
    [0, 0, 0, 0, 3, 3, 5, 5],
    [0, 0, 0, 0, 3, 3, 5, 5],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [2, 0, 0, 7, 8],
    [2, 1, 1, 0, 0],
    [0, 5, 6, 6, 0],
    [3, 5, 6, 0, 0],
    [0, 5, 0, 0, 0],
], dtype=int)

T_OUT = np.array([
    [2, 2, 0, 0, 0, 0, 7, 7, 8, 8],
    [2, 2, 0, 0, 0, 0, 7, 7, 8, 8],
    [2, 2, 1, 1, 1, 1, 0, 0, 0, 0],
    [2, 2, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 5, 5, 6, 6, 6, 6, 0, 0],
    [0, 0, 5, 5, 6, 6, 6, 6, 0, 0],
    [3, 3, 5, 5, 6, 6, 0, 0, 0, 0],
    [3, 3, 5, 5, 6, 6, 0, 0, 0, 0],
    [0, 0, 5, 5, 0, 0, 0, 0, 0, 0],
    [0, 0, 5, 5, 0, 0, 0, 0, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(g):
 X=[]
 for r in g:
  for i in range(2):
   X+=[sum([[c]*2 for c in r],[])]
 return X


# --- Code Golf Solution (Compressed) ---
def q(g):
    return g * -1 * -1 or (g and [p(g[0])] * 2 + p(g[1:]))


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

Element = Union[Object, Grid]

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

def upscale(
    element: Element,
    factor: Integer
) -> Element:
    """ upscale object or grid """
    if isinstance(element, tuple):
        upscaled_grid = tuple()
        for row in element:
            upscaled_row = tuple()
            for value in row:
                upscaled_row = upscaled_row + tuple(value for num in range(factor))
            upscaled_grid = upscaled_grid + tuple(upscaled_row for num in range(factor))
        return upscaled_grid
    else:
        if len(element) == 0:
            return frozenset()
        di_inv, dj_inv = ulcorner(element)
        di, dj = (-di_inv, -dj_inv)
        normed_obj = shift(element, (di, dj))
        upscaled_obj = set()
        for value, (i, j) in normed_obj:
            for io in range(factor):
                for jo in range(factor):
                    upscaled_obj.add((value, (i * factor + io, j * factor + jo)))
        return shift(frozenset(upscaled_obj), (di_inv, dj_inv))

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

def generate_c59eb873(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 15))
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
    go = upscale(gi, 2)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
TWO = 2

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_c59eb873(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = upscale(I, TWO)
    return x0


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_c59eb873(inp)
        assert pred == _to_grid(expected), f"{name} failed"

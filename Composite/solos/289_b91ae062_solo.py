# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "b91ae062"
SERIAL = "289"
URL    = "https://arcprize.org/play?task=b91ae062"

# --- Code Golf Concepts ---
CONCEPTS = [
    "image_resizing",
    "size_guessing",
    "count_different_colors",
]

# --- Example Grids ---
E1_IN = np.array([
    [6, 7, 0],
    [0, 6, 6],
    [0, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [6, 6, 7, 7, 0, 0],
    [6, 6, 7, 7, 0, 0],
    [0, 0, 6, 6, 6, 6],
    [0, 0, 6, 6, 6, 6],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
], dtype=int)

E2_IN = np.array([
    [1, 0, 4],
    [0, 4, 0],
    [0, 1, 0],
], dtype=int)

E2_OUT = np.array([
    [1, 1, 0, 0, 4, 4],
    [1, 1, 0, 0, 4, 4],
    [0, 0, 4, 4, 0, 0],
    [0, 0, 4, 4, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0],
], dtype=int)

E3_IN = np.array([
    [3, 2, 0],
    [0, 7, 3],
    [0, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [3, 3, 3, 2, 2, 2, 0, 0, 0],
    [3, 3, 3, 2, 2, 2, 0, 0, 0],
    [3, 3, 3, 2, 2, 2, 0, 0, 0],
    [0, 0, 0, 7, 7, 7, 3, 3, 3],
    [0, 0, 0, 7, 7, 7, 3, 3, 3],
    [0, 0, 0, 7, 7, 7, 3, 3, 3],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E4_IN = np.array([
    [0, 8, 0],
    [0, 6, 6],
    [9, 8, 0],
], dtype=int)

E4_OUT = np.array([
    [0, 0, 0, 8, 8, 8, 0, 0, 0],
    [0, 0, 0, 8, 8, 8, 0, 0, 0],
    [0, 0, 0, 8, 8, 8, 0, 0, 0],
    [0, 0, 0, 6, 6, 6, 6, 6, 6],
    [0, 0, 0, 6, 6, 6, 6, 6, 6],
    [0, 0, 0, 6, 6, 6, 6, 6, 6],
    [9, 9, 9, 8, 8, 8, 0, 0, 0],
    [9, 9, 9, 8, 8, 8, 0, 0, 0],
    [9, 9, 9, 8, 8, 8, 0, 0, 0],
], dtype=int)

E5_IN = np.array([
    [4, 0, 3],
    [2, 2, 0],
    [0, 0, 8],
], dtype=int)

E5_OUT = np.array([
    [4, 4, 4, 4, 0, 0, 0, 0, 3, 3, 3, 3],
    [4, 4, 4, 4, 0, 0, 0, 0, 3, 3, 3, 3],
    [4, 4, 4, 4, 0, 0, 0, 0, 3, 3, 3, 3],
    [4, 4, 4, 4, 0, 0, 0, 0, 3, 3, 3, 3],
    [2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0],
    [2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0],
    [2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0],
    [2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8],
    [0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8],
    [0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8],
    [0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 1, 0],
    [0, 8, 7],
    [9, 9, 0],
], dtype=int)

T_OUT = np.array([
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 8, 8, 8, 7, 7, 7, 7],
    [0, 0, 0, 0, 8, 8, 8, 8, 7, 7, 7, 7],
    [0, 0, 0, 0, 8, 8, 8, 8, 7, 7, 7, 7],
    [0, 0, 0, 0, 8, 8, 8, 8, 7, 7, 7, 7],
    [9, 9, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0],
    [9, 9, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0],
    [9, 9, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0],
    [9, 9, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
    return ((A := len(set(sum(j, [])) - {0})), [[x for x in r for _ in range(A)] for r in j for _ in range(A)])[1]


# --- Code Golf Solution (Compressed) ---
def q(g):
    return eval("[[g\nfor g in g for _ in[*{*'%r'}][5:]]#" % g * 2)


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

def generate_b91ae062(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 5))
    w = unifint(diff_lb, diff_ub, (2, 5))
    numc = unifint(diff_lb, diff_ub, (3, min(h * w, min(10, 30 // max(h, w)))))
    ccols = sample(cols, numc)
    c = canvas(-1, (h, w))
    inds = totuple(asindices(c))
    fixinds = sample(inds, numc)
    obj = {(cc, ij) for cc, ij in zip(ccols, fixinds)}
    for ij in difference(inds, fixinds):
        obj.add((choice(ccols), ij))
    gi = paint(c, obj)
    go = upscale(gi, numc - 1)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Numerical = Union[Integer, IntegerTuple]

IntegerSet = FrozenSet[Integer]

def decrement(
    x: Numerical
) -> Numerical:
    """ decrementing """
    return x - 1 if isinstance(x, int) else (x[0] - 1, x[1] - 1)

def palette(
    element: Element
) -> IntegerSet:
    """ colors occurring in object or grid """
    if isinstance(element, tuple):
        return frozenset({v for r in element for v in r})
    return frozenset({v for v, _ in element})

def numcolors(
    element: Element
) -> IntegerSet:
    """ number of colors occurring in object or grid """
    return len(palette(element))

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_b91ae062(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = numcolors(I)
    x1 = decrement(x0)
    x2 = upscale(I, x1)
    return x2


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("E4", E4_IN, E4_OUT),
        ("E5", E5_IN, E5_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_b91ae062(inp)
        assert pred == _to_grid(expected), f"{name} failed"

# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "ac0a08a4"
SERIAL = "269"
URL    = "https://arcprize.org/play?task=ac0a08a4"

# --- Code Golf Concepts ---
CONCEPTS = [
    "image_resizing",
    "count_tiles",
    "size_guessing",
]

# --- Example Grids ---
E1_IN = np.array([
    [2, 0, 0],
    [0, 0, 7],
    [0, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [2, 2, 0, 0, 0, 0],
    [2, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 7, 7],
    [0, 0, 0, 0, 7, 7],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
], dtype=int)

E2_IN = np.array([
    [0, 4, 0],
    [0, 0, 8],
    [6, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [0, 0, 0, 4, 4, 4, 0, 0, 0],
    [0, 0, 0, 4, 4, 4, 0, 0, 0],
    [0, 0, 0, 4, 4, 4, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 8, 8, 8],
    [0, 0, 0, 0, 0, 0, 8, 8, 8],
    [0, 0, 0, 0, 0, 0, 8, 8, 8],
    [6, 6, 6, 0, 0, 0, 0, 0, 0],
    [6, 6, 6, 0, 0, 0, 0, 0, 0],
    [6, 6, 6, 0, 0, 0, 0, 0, 0],
], dtype=int)

E3_IN = np.array([
    [0, 6, 9],
    [3, 0, 2],
    [0, 7, 0],
], dtype=int)

E3_OUT = np.array([
    [0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 9, 9, 9, 9, 9],
    [0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 9, 9, 9, 9, 9],
    [0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 9, 9, 9, 9, 9],
    [0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 9, 9, 9, 9, 9],
    [0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 9, 9, 9, 9, 9],
    [3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2],
    [3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2],
    [3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2],
    [3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2],
    [3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2],
    [0, 0, 0, 0, 0, 7, 7, 7, 7, 7, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 7, 7, 7, 7, 7, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 7, 7, 7, 7, 7, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 7, 7, 7, 7, 7, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 7, 7, 7, 7, 7, 0, 0, 0, 0, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [1, 0, 0],
    [0, 9, 6],
    [8, 0, 0],
], dtype=int)

T_OUT = np.array([
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 9, 9, 9, 9, 6, 6, 6, 6],
    [0, 0, 0, 0, 9, 9, 9, 9, 6, 6, 6, 6],
    [0, 0, 0, 0, 9, 9, 9, 9, 6, 6, 6, 6],
    [0, 0, 0, 0, 9, 9, 9, 9, 6, 6, 6, 6],
    [8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
    return ((A := sum((c > 0 for r in j for c in r))), [sum(([x] * A for x in r), []) for r in j for _ in range(A)])[1]


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

def generate_ac0a08a4(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 5))
    w = unifint(diff_lb, diff_ub, (2, 5))
    num = unifint(diff_lb, diff_ub, (1, min(min(9, h * w - 2), min(30//h, 30//w))))
    bgc = choice(cols)
    c = canvas(bgc, (h, w))
    inds = asindices(c)
    locs = sample(totuple(inds), num)
    remcols = remove(bgc, cols)
    obj = {(col, loc) for col, loc in zip(sample(remcols, num), locs)}
    gi = paint(c, obj)
    go = upscale(gi, num)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Numerical = Union[Integer, IntegerTuple]

Piece = Union[Grid, Patch]

def subtract(
    a: Numerical,
    b: Numerical
) -> Numerical:
    """ subtraction """
    if isinstance(a, int) and isinstance(b, int):
        return a - b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] - b[0], a[1] - b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a - b[0], a - b[1])
    return (a[0] - b, a[1] - b)

def multiply(
    a: Numerical,
    b: Numerical
) -> Numerical:
    """ multiplication """
    if isinstance(a, int) and isinstance(b, int):
        return a * b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] * b[0], a[1] * b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a * b[0], a * b[1])
    return (a[0] * b, a[1] * b)

def mostcolor(
    element: Element
) -> Integer:
    """ most common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return max(set(values), key=values.count)

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

def colorcount(
    element: Element,
    value: Integer
) -> Integer:
    """ number of cells with color """
    if isinstance(element, tuple):
        return sum(row.count(value) for row in element)
    return sum(v == value for v, _ in element)

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

def verify_ac0a08a4(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = mostcolor(I)
    x1 = colorcount(I, x0)
    x2 = height(I)
    x3 = width(I)
    x4 = multiply(x2, x3)
    x5 = subtract(x4, x1)
    x6 = upscale(I, x5)
    return x6


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_ac0a08a4(inp)
        assert pred == _to_grid(expected), f"{name} failed"

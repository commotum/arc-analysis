# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "cf98881b"
SERIAL = "321"
URL    = "https://arcprize.org/play?task=cf98881b"

# --- Code Golf Concepts ---
CONCEPTS = [
    "detect_wall",
    "separate_images",
    "pattern_juxtaposition",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 4, 0, 4, 2, 9, 9, 0, 0, 2, 0, 0, 0, 0],
    [0, 4, 0, 0, 2, 0, 0, 9, 9, 2, 0, 1, 0, 0],
    [4, 0, 0, 0, 2, 0, 0, 0, 0, 2, 1, 1, 1, 0],
    [4, 4, 4, 4, 2, 9, 0, 9, 0, 2, 1, 1, 0, 1],
], dtype=int)

E1_OUT = np.array([
    [9, 4, 0, 4],
    [0, 4, 9, 9],
    [4, 1, 1, 0],
    [4, 4, 4, 4],
], dtype=int)

E2_IN = np.array([
    [4, 4, 4, 4, 2, 9, 0, 9, 0, 2, 0, 0, 0, 1],
    [4, 4, 0, 0, 2, 9, 9, 0, 0, 2, 1, 0, 0, 0],
    [4, 0, 4, 4, 2, 0, 0, 0, 9, 2, 0, 1, 0, 1],
    [0, 0, 0, 0, 2, 0, 0, 9, 0, 2, 1, 0, 1, 0],
], dtype=int)

E2_OUT = np.array([
    [4, 4, 4, 4],
    [4, 4, 0, 0],
    [4, 1, 4, 4],
    [1, 0, 9, 0],
], dtype=int)

E3_IN = np.array([
    [4, 4, 4, 0, 2, 9, 9, 0, 9, 2, 0, 1, 0, 1],
    [0, 4, 0, 4, 2, 0, 0, 9, 0, 2, 0, 1, 0, 0],
    [0, 4, 0, 4, 2, 0, 0, 9, 9, 2, 1, 0, 0, 1],
    [4, 0, 4, 4, 2, 9, 9, 9, 0, 2, 0, 0, 0, 1],
], dtype=int)

E3_OUT = np.array([
    [4, 4, 4, 9],
    [0, 4, 9, 4],
    [1, 4, 9, 4],
    [4, 9, 4, 4],
], dtype=int)

E4_IN = np.array([
    [0, 0, 0, 4, 2, 0, 0, 0, 9, 2, 0, 0, 0, 0],
    [4, 4, 0, 4, 2, 9, 0, 9, 0, 2, 0, 0, 0, 0],
    [4, 0, 4, 4, 2, 0, 9, 9, 0, 2, 1, 1, 0, 1],
    [0, 4, 4, 4, 2, 0, 9, 0, 0, 2, 1, 1, 1, 1],
], dtype=int)

E4_OUT = np.array([
    [0, 0, 0, 4],
    [4, 4, 9, 4],
    [4, 9, 4, 4],
    [1, 4, 4, 4],
], dtype=int)

E5_IN = np.array([
    [4, 0, 4, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 1],
    [4, 4, 4, 4, 2, 0, 0, 0, 9, 2, 1, 1, 0, 0],
    [0, 4, 4, 4, 2, 0, 9, 9, 0, 2, 1, 1, 0, 1],
    [0, 4, 4, 0, 2, 0, 0, 9, 0, 2, 0, 1, 0, 1],
], dtype=int)

E5_OUT = np.array([
    [4, 0, 4, 1],
    [4, 4, 4, 4],
    [1, 4, 4, 4],
    [0, 4, 4, 1],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 4, 0, 2, 9, 0, 9, 0, 2, 1, 1, 0, 0],
    [4, 4, 0, 4, 2, 9, 9, 9, 0, 2, 1, 1, 1, 0],
    [0, 0, 0, 0, 2, 0, 9, 9, 9, 2, 1, 1, 0, 1],
    [0, 4, 4, 0, 2, 9, 0, 9, 9, 2, 1, 0, 0, 0],
], dtype=int)

T_OUT = np.array([
    [9, 1, 4, 0],
    [4, 4, 9, 4],
    [1, 9, 9, 9],
    [9, 4, 4, 9],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
 for A in range(4):
  for c in range(4):
   if j[A][c+5]>0:j[A][c+10]=j[A][c+5]
   if j[A][c]>0:j[A][c+10]=j[A][c]
 return[R[10:]for R in j]


# --- Code Golf Solution (Compressed) ---
def q(g):
    return [eval('r.pop(0)or r[4]|r[9],' * 4) for r in g]


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

def generate_cf98881b(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 9))
    bgc, barcol, cola, colb, colc = sample(cols, 5)
    canv = canvas(bgc, (h, w))
    inds = totuple(asindices(canv))
    gbar = canvas(barcol, (h, 1))
    mp = (h * w) // 2
    devrng = (0, mp)
    deva = unifint(diff_lb, diff_ub, devrng)
    devb = unifint(diff_lb, diff_ub, devrng)
    devc = unifint(diff_lb, diff_ub, devrng)
    sgna = choice((+1, -1))
    sgnb = choice((+1, -1))
    sgnc = choice((+1, -1))
    deva = sgna * deva
    devb = sgnb * devb
    devc = sgnc * devc
    numa = mp + deva
    numb = mp + devb
    numc = mp + devc
    numa = max(min(h * w - 1, numa), 1)
    numb = max(min(h * w - 1, numb), 1)
    numc = max(min(h * w - 1, numc), 1)
    a = sample(inds, numa)
    b = sample(inds, numb)
    c = sample(inds, numc)
    gia = fill(canv, cola, a)
    gib = fill(canv, colb, b)
    gic = fill(canv, colc, c)
    gi = hconcat(hconcat(hconcat(gia, gbar), hconcat(gib, gbar)), gic)
    go = fill(gic, colb, b)
    go = fill(go, cola, a)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Numerical = Union[Integer, IntegerTuple]

IntegerSet = FrozenSet[Integer]

Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

ONE = 1

TWO = 2

THREE = 3

ORIGIN = (0, 0)

def add(
    a: Numerical,
    b: Numerical
) -> Numerical:
    """ addition """
    if isinstance(a, int) and isinstance(b, int):
        return a + b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] + b[0], a[1] + b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a + b[0], a + b[1])
    return (a[0] + b, a[1] + b)

def divide(
    a: Numerical,
    b: Numerical
) -> Numerical:
    """ floor division """
    if isinstance(a, int) and isinstance(b, int):
        return a // b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] // b[0], a[1] // b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a // b[0], a // b[1])
    return (a[0] // b, a[1] // b)

def double(
    n: Numerical
) -> Numerical:
    """ scaling by two """
    return n * 2 if isinstance(n, int) else (n[0] * 2, n[1] * 2)

def intersection(
    a: FrozenSet,
    b: FrozenSet
) -> FrozenSet:
    """ returns the intersection of two containers """
    return a & b

def increment(
    x: Numerical
) -> Numerical:
    """ incrementing """
    return x + 1 if isinstance(x, int) else (x[0] + 1, x[1] + 1)

def decrement(
    x: Numerical
) -> Numerical:
    """ decrementing """
    return x - 1 if isinstance(x, int) else (x[0] - 1, x[1] - 1)

def tojvec(
    j: Integer
) -> IntegerTuple:
    """ vector pointing horizontally """
    return (0, j)

def first(
    container: Container
) -> Any:
    """ first item of container """
    return next(iter(container))

def remove(
    value: Any,
    container: Container
) -> Container:
    """ remove item from container """
    return type(container)(e for e in container if e != value)

def other(
    container: Container,
    value: Any
) -> Any:
    """ other value in the container """
    return first(remove(value, container))

def astuple(
    a: Integer,
    b: Integer
) -> IntegerTuple:
    """ constructs a tuple """
    return (a, b)

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

def ofcolor(
    grid: Grid,
    value: Integer
) -> Indices:
    """ indices of all grid cells with value """
    return frozenset((i, j) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value)

def crop(
    grid: Grid,
    start: IntegerTuple,
    dims: IntegerTuple
) -> Grid:
    """ subgrid specified by start and dimension """
    return tuple(r[start[1]:start[1]+dims[1]] for r in grid[start[0]:start[0]+dims[0]])

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

def palette(
    element: Element
) -> IntegerSet:
    """ colors occurring in object or grid """
    if isinstance(element, tuple):
        return frozenset({v for r in element for v in r})
    return frozenset({v for v, _ in element})

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_cf98881b(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = width(I)
    x1 = increment(x0)
    x2 = divide(x1, THREE)
    x3 = decrement(x2)
    x4 = height(I)
    x5 = astuple(x4, x3)
    x6 = crop(I, ORIGIN, x5)
    x7 = add(x3, ONE)
    x8 = tojvec(x7)
    x9 = crop(I, x8, x5)
    x10 = double(x3)
    x11 = add(x10, TWO)
    x12 = tojvec(x11)
    x13 = crop(I, x12, x5)
    x14 = palette(x6)
    x15 = palette(x9)
    x16 = palette(x13)
    x17 = intersection(x14, x15)
    x18 = intersection(x17, x16)
    x19 = first(x18)
    x20 = other(x14, x19)
    x21 = other(x15, x19)
    x22 = other(x16, x19)
    x23 = canvas(x19, x5)
    x24 = ofcolor(x6, x20)
    x25 = ofcolor(x9, x21)
    x26 = ofcolor(x13, x22)
    x27 = fill(x23, x22, x26)
    x28 = fill(x27, x21, x25)
    x29 = fill(x28, x20, x24)
    return x29


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
        pred = verify_cf98881b(inp)
        assert pred == _to_grid(expected), f"{name} failed"

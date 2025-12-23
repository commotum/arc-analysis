# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "d23f8c26"
SERIAL = "329"
URL    = "https://arcprize.org/play?task=d23f8c26"

# --- Code Golf Concepts ---
CONCEPTS = [
    "crop",
    "image_expansion",
]

# --- Example Grids ---
E1_IN = np.array([
    [6, 4, 0],
    [0, 3, 9],
    [1, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [0, 4, 0],
    [0, 3, 0],
    [0, 0, 0],
], dtype=int)

E2_IN = np.array([
    [8, 0, 3, 0, 0],
    [8, 6, 5, 6, 0],
    [3, 6, 3, 0, 0],
    [0, 0, 0, 5, 9],
    [5, 0, 9, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [0, 0, 3, 0, 0],
    [0, 0, 5, 0, 0],
    [0, 0, 3, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 9, 0, 0],
], dtype=int)

E3_IN = np.array([
    [3, 0, 4, 0, 0],
    [3, 0, 4, 7, 0],
    [0, 6, 0, 0, 7],
    [0, 0, 8, 0, 0],
    [0, 8, 0, 2, 2],
], dtype=int)

E3_OUT = np.array([
    [0, 0, 4, 0, 0],
    [0, 0, 4, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 8, 0, 0],
    [0, 0, 0, 0, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 3, 0, 0, 0, 7],
    [8, 1, 0, 8, 0, 0, 0],
    [0, 0, 3, 0, 8, 0, 3],
    [0, 7, 0, 1, 0, 7, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [1, 0, 8, 6, 0, 0, 0],
    [0, 8, 0, 6, 0, 1, 0],
], dtype=int)

T_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 6, 0, 0, 0],
    [0, 0, 0, 6, 0, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
	A=len(j[0])//2;c=[[0 for A in A]for A in j]
	for E in range(len(j)):c[E][A]=j[E][A]
	return c


# --- Code Golf Solution (Compressed) ---
def q(m):
    return [(n := (len(r) // 2)) * [0] + [r[n]] + n * [0] for r in m]


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

def generate_d23f8c26(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (2, 30))
    wh = unifint(diff_lb, diff_ub, (1, 14))
    w = 2 * wh + 1
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    numn = unifint(diff_lb, diff_ub, (1, (h * w) // 2 - 1))
    numcols = unifint(diff_lb, diff_ub, (1, 9))
    remcols = sample(remcols, numcols)
    inds = totuple(asindices(gi))
    locs = sample(inds, numn)
    for ij in locs:
        col = choice(remcols)
        gi = fill(gi, col, {ij})
        a, b = ij
        if b == w // 2:
            go = fill(go, col, {ij})
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Numerical = Union[Integer, IntegerTuple]

Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

def halve(
    n: Numerical
) -> Numerical:
    """ scaling by one half """
    return n // 2 if isinstance(n, int) else (n[0] // 2, n[1] // 2)

def flip(
    b: Boolean
) -> Boolean:
    """ logical not """
    return not b

def both(
    a: Boolean,
    b: Boolean
) -> Boolean:
    """ logical and """
    return a and b

def sfilter(
    container: Container,
    condition: Callable
) -> Container:
    """ keep elements in container that satisfy condition """
    return type(container)(e for e in container if condition(e))

def first(
    container: Container
) -> Any:
    """ first item of container """
    return next(iter(container))

def last(
    container: Container
) -> Any:
    """ last item of container """
    return max(enumerate(container))[1]

def compose(
    outer: Callable,
    inner: Callable
) -> Callable:
    """ function composition """
    return lambda x: outer(inner(x))

def matcher(
    function: Callable,
    target: Any
) -> Callable:
    """ construction of equality function """
    return lambda x: function(x) == target

def fork(
    outer: Callable,
    a: Callable,
    b: Callable
) -> Callable:
    """ creates a wrapper function """
    return lambda x: outer(a(x), b(x))

def mostcolor(
    element: Element
) -> Integer:
    """ most common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return max(set(values), key=values.count)

def width(
    piece: Piece
) -> Integer:
    """ width of grid or patch """
    if len(piece) == 0:
        return 0
    if isinstance(piece, tuple):
        return len(piece[0])
    return rightmost(piece) - leftmost(piece) + 1

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

def asobject(
    grid: Grid
) -> Object:
    """ conversion of grid to object """
    return frozenset((v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r))

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_d23f8c26(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = mostcolor(I)
    x1 = matcher(first, x0)
    x2 = compose(flip, x1)
    x3 = width(I)
    x4 = halve(x3)
    x5 = compose(last, last)
    x6 = matcher(x5, x4)
    x7 = compose(flip, x6)
    x8 = asobject(I)
    x9 = fork(both, x2, x7)
    x10 = sfilter(x8, x9)
    x11 = fill(I, x0, x10)
    return x11


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_d23f8c26(inp)
        assert pred == _to_grid(expected), f"{name} failed"

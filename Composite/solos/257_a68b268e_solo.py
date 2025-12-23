# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "a68b268e"
SERIAL = "257"
URL    = "https://arcprize.org/play?task=a68b268e"

# --- Code Golf Concepts ---
CONCEPTS = [
    "detect_grid",
    "separate_images",
    "pattern_juxtaposition",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 7, 7, 7, 1, 0, 4, 0, 4],
    [7, 7, 7, 0, 1, 4, 4, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 4],
    [7, 0, 0, 0, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 6, 6, 6, 0],
    [0, 0, 8, 8, 1, 0, 0, 0, 0],
    [8, 0, 8, 0, 1, 6, 0, 0, 6],
    [0, 0, 0, 8, 1, 0, 0, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [6, 7, 7, 7],
    [7, 7, 7, 8],
    [8, 0, 8, 4],
    [7, 0, 0, 8],
], dtype=int)

E2_IN = np.array([
    [7, 7, 7, 0, 1, 0, 4, 0, 0],
    [7, 0, 7, 0, 1, 4, 0, 4, 4],
    [0, 7, 0, 7, 1, 4, 0, 4, 4],
    [0, 0, 0, 7, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 8, 0, 1, 6, 0, 0, 6],
    [0, 0, 0, 0, 1, 6, 0, 0, 0],
    [0, 0, 0, 0, 1, 6, 6, 0, 6],
    [8, 8, 8, 0, 1, 6, 0, 6, 6],
], dtype=int)

E2_OUT = np.array([
    [7, 7, 7, 6],
    [7, 0, 7, 4],
    [4, 7, 4, 7],
    [8, 8, 8, 7],
], dtype=int)

E3_IN = np.array([
    [0, 0, 7, 7, 1, 0, 4, 4, 0],
    [0, 0, 0, 7, 1, 0, 0, 4, 4],
    [7, 7, 7, 7, 1, 0, 0, 0, 4],
    [0, 7, 0, 0, 1, 0, 4, 4, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 8, 8, 1, 0, 6, 6, 6],
    [0, 0, 0, 0, 1, 0, 0, 6, 0],
    [0, 0, 0, 8, 1, 6, 0, 6, 0],
    [8, 0, 0, 0, 1, 6, 6, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [0, 4, 7, 7],
    [0, 0, 4, 7],
    [7, 7, 7, 7],
    [8, 7, 4, 0],
], dtype=int)

E4_IN = np.array([
    [7, 7, 0, 0, 1, 4, 4, 0, 4],
    [7, 0, 7, 0, 1, 4, 0, 0, 0],
    [7, 0, 0, 7, 1, 4, 4, 4, 0],
    [7, 0, 7, 7, 1, 4, 0, 4, 4],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 8, 0, 1, 0, 0, 0, 0],
    [0, 0, 8, 0, 1, 6, 6, 0, 0],
    [0, 0, 8, 0, 1, 0, 6, 6, 6],
    [0, 8, 0, 8, 1, 0, 6, 6, 0],
], dtype=int)

E4_OUT = np.array([
    [7, 7, 8, 4],
    [7, 6, 7, 0],
    [7, 4, 4, 7],
    [7, 8, 7, 7],
], dtype=int)

E5_IN = np.array([
    [7, 7, 0, 0, 1, 0, 0, 0, 4],
    [7, 0, 0, 0, 1, 4, 4, 4, 4],
    [7, 0, 7, 0, 1, 4, 0, 0, 0],
    [0, 7, 7, 0, 1, 4, 4, 4, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [8, 0, 8, 0, 1, 6, 6, 6, 6],
    [0, 0, 8, 8, 1, 0, 0, 6, 0],
    [0, 0, 0, 0, 1, 0, 6, 0, 6],
    [8, 8, 8, 8, 1, 0, 0, 0, 6],
], dtype=int)

E5_OUT = np.array([
    [7, 7, 8, 4],
    [7, 4, 4, 4],
    [7, 6, 7, 6],
    [4, 7, 7, 8],
], dtype=int)

E6_IN = np.array([
    [7, 0, 0, 7, 1, 4, 4, 4, 0],
    [0, 7, 7, 7, 1, 4, 4, 0, 4],
    [7, 7, 7, 0, 1, 4, 4, 0, 4],
    [7, 7, 7, 0, 1, 0, 4, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [8, 8, 0, 8, 1, 6, 6, 6, 6],
    [0, 8, 8, 8, 1, 0, 0, 0, 6],
    [0, 8, 0, 8, 1, 0, 0, 6, 0],
    [8, 8, 0, 8, 1, 0, 6, 0, 0],
], dtype=int)

E6_OUT = np.array([
    [7, 4, 4, 7],
    [4, 7, 7, 7],
    [7, 7, 7, 4],
    [7, 7, 7, 8],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [7, 7, 7, 0, 1, 0, 0, 4, 0],
    [0, 7, 7, 0, 1, 4, 4, 0, 4],
    [7, 7, 7, 7, 1, 0, 4, 0, 4],
    [7, 0, 0, 0, 1, 4, 0, 4, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 8, 1, 0, 6, 0, 6],
    [8, 0, 0, 8, 1, 6, 0, 0, 6],
    [8, 0, 8, 0, 1, 6, 6, 6, 6],
    [0, 8, 0, 8, 1, 0, 6, 0, 0],
], dtype=int)

T_OUT = np.array([
    [7, 7, 7, 8],
    [4, 7, 7, 4],
    [7, 7, 7, 7],
    [7, 8, 4, 8],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(g,L=len,R=range):
 h,w=L(g),L(g[0])
 for r in R(4):
  for c in R(4):
   if g[r][c]==0:
    if g[r][c+5]>0:g[r][c]=g[r][c+5]
   if g[r][c]==0:
    if g[r+5][c]>0:g[r][c]=g[r+5][c]
   if g[r][c]==0:
    if g[r+5][c+5]>0:g[r][c]=g[r+5][c+5]
 return [r[:4] for r in g[:4]]


# --- Code Golf Solution (Compressed) ---
def q(a):
    return [p(b) for *b, in map(zip, a, a[5:])] or max(sum(a, ()), key=bool)


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

def generate_a68b268e(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 14))
    w = unifint(diff_lb, diff_ub, (2, 4))
    bgc, linc, c1, c2, c3, c4 = sample(cols, 6)
    canv = canvas(bgc, (h, w))
    inds = asindices(canv)
    nc1d = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    nc1 = choice((nc1d, h * w - nc1d))
    nc1 = min(max(1, nc1), h * w - 1)
    nc2d = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    nc2 = choice((nc2d, h * w - nc2d))
    nc2 = min(max(1, nc2), h * w - 1)
    nc3d = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    nc3 = choice((nc3d, h * w - nc3d))
    nc3 = min(max(1, nc3), h * w - 1)
    nc4d = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    nc4 = choice((nc4d, h * w - nc4d))
    nc4 = min(max(1, nc4), h * w - 1)
    ofc1 = sample(totuple(inds), nc1)
    ofc2 = sample(totuple(inds), nc2)
    ofc3 = sample(totuple(inds), nc3)
    ofc4 = sample(totuple(inds), nc4)
    go = fill(canv, c1, ofc1)
    go = fill(go, c2, ofc2)
    go = fill(go, c3, ofc3)
    go = fill(go, c4, ofc4)
    LR = asobject(fill(canv, c1, ofc1))
    LL = asobject(fill(canv, c2, ofc2))
    UR = asobject(fill(canv, c3, ofc3))
    UL = asobject(fill(canv, c4, ofc4))
    gi = canvas(linc, (2*h+1, 2*w+1))
    gi = paint(gi, shift(LR, (h+1, w+1)))
    gi = paint(gi, shift(LL, (h+1, 0)))
    gi = paint(gi, shift(UR, (0, w+1)))
    gi = paint(gi, shift(UL, (0, 0)))
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Numerical = Union[Integer, IntegerTuple]

IntegerSet = FrozenSet[Integer]

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

def intersection(
    a: FrozenSet,
    b: FrozenSet
) -> FrozenSet:
    """ returns the intersection of two containers """
    return a & b

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

def rbind(
    function: Callable,
    fixed: Any
) -> Callable:
    """ fix the rightmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda x: function(x, fixed)
    elif n == 3:
        return lambda x, y: function(x, y, fixed)
    else:
        return lambda x, y, z: function(x, y, z, fixed)

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

def tophalf(
    grid: Grid
) -> Grid:
    """ upper half of grid """
    return grid[:len(grid) // 2]

def bottomhalf(
    grid: Grid
) -> Grid:
    """ lower half of grid """
    return grid[len(grid) // 2 + len(grid) % 2:]

def lefthalf(
    grid: Grid
) -> Grid:
    """ left half of grid """
    return rot270(tophalf(rot90(grid)))

def righthalf(
    grid: Grid
) -> Grid:
    """ right half of grid """
    return rot270(bottomhalf(rot90(grid)))

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_a68b268e(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = tophalf(I)
    x1 = lefthalf(x0)
    x2 = tophalf(I)
    x3 = righthalf(x2)
    x4 = bottomhalf(I)
    x5 = lefthalf(x4)
    x6 = bottomhalf(I)
    x7 = righthalf(x6)
    x8 = palette(x1)
    x9 = palette(x3)
    x10 = intersection(x8, x9)
    x11 = palette(x5)
    x12 = palette(x7)
    x13 = intersection(x11, x12)
    x14 = intersection(x10, x13)
    x15 = first(x14)
    x16 = shape(I)
    x17 = halve(x16)
    x18 = canvas(x15, x17)
    x19 = matcher(first, x15)
    x20 = compose(flip, x19)
    x21 = rbind(sfilter, x20)
    x22 = compose(x21, asobject)
    x23 = x22(x1)
    x24 = x22(x3)
    x25 = x22(x5)
    x26 = x22(x7)
    x27 = paint(x18, x26)
    x28 = paint(x27, x25)
    x29 = paint(x28, x24)
    x30 = paint(x29, x23)
    return x30


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("E4", E4_IN, E4_OUT),
        ("E5", E5_IN, E5_OUT),
        ("E6", E6_IN, E6_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_a68b268e(inp)
        assert pred == _to_grid(expected), f"{name} failed"

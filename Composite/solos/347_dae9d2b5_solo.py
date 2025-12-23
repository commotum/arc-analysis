# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "dae9d2b5"
SERIAL = "347"
URL    = "https://arcprize.org/play?task=dae9d2b5"

# --- Code Golf Concepts ---
CONCEPTS = [
    "pattern_juxtaposition",
    "separate_images",
    "recoloring",
]

# --- Example Grids ---
E1_IN = np.array([
    [4, 4, 0, 3, 3, 0],
    [4, 0, 0, 3, 0, 0],
    [0, 0, 0, 0, 0, 3],
], dtype=int)

E1_OUT = np.array([
    [6, 6, 0],
    [6, 0, 0],
    [0, 0, 6],
], dtype=int)

E2_IN = np.array([
    [4, 0, 4, 3, 3, 0],
    [4, 0, 0, 3, 0, 0],
    [0, 0, 4, 3, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [6, 6, 6],
    [6, 0, 0],
    [6, 0, 6],
], dtype=int)

E3_IN = np.array([
    [0, 0, 4, 0, 3, 0],
    [0, 4, 4, 3, 0, 3],
    [4, 4, 0, 0, 0, 3],
], dtype=int)

E3_OUT = np.array([
    [0, 6, 6],
    [6, 6, 6],
    [6, 6, 6],
], dtype=int)

E4_IN = np.array([
    [4, 4, 0, 3, 0, 0],
    [0, 0, 0, 0, 0, 3],
    [4, 0, 0, 0, 0, 0],
], dtype=int)

E4_OUT = np.array([
    [6, 6, 0],
    [0, 0, 6],
    [6, 0, 0],
], dtype=int)

E5_IN = np.array([
    [0, 0, 0, 0, 3, 0],
    [4, 0, 0, 0, 0, 0],
    [0, 0, 4, 3, 3, 0],
], dtype=int)

E5_OUT = np.array([
    [0, 6, 0],
    [6, 0, 0],
    [6, 6, 6],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 4, 4, 3, 0, 0],
    [4, 0, 0, 3, 3, 0],
    [0, 4, 0, 3, 0, 0],
], dtype=int)

T_OUT = np.array([
    [6, 6, 6],
    [6, 6, 0],
    [6, 6, 0],
], dtype=int)

T2_IN = np.array([
    [0, 0, 4, 0, 3, 0],
    [0, 4, 0, 3, 3, 3],
    [4, 0, 0, 3, 0, 0],
], dtype=int)

T2_OUT = np.array([
    [0, 6, 6],
    [6, 6, 6],
    [6, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j,A=range(3)):
 for c in A:
  for E in A:
   j[c][E]+=j[c][E+3]
   if j[c][E]>0:j[c][E]=6
 return[R[:3]for R in j]


# --- Code Golf Solution (Compressed) ---
def q(m):
    return [[6 * (-r.pop(3) < a) for a in r] for r in m]


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

def dmirror(
    piece: Piece
) -> Piece:
    """ mirroring along diagonal """
    if isinstance(piece, tuple):
        return tuple(zip(*piece))
    a, b = ulcorner(piece)
    if isinstance(next(iter(piece))[1], tuple):
        return frozenset((v, (j - b + a, i - a + b)) for v, (i, j) in piece)
    return frozenset((j - b + a, i - a + b) for i, j in piece)

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

def generate_dae9d2b5(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(6, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 14))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    acol = choice(remcols)
    remcols = remove(acol, remcols)
    bcol = choice(remcols)
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    numadev = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numbdev = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numa = choice((numadev, h * w - numadev))
    numb = choice((numadev, h * w - numbdev))
    numa = min(max(1, numa), h * w - 1)
    numb = min(max(1, numb), h * w - 1)
    aset = sample(inds, numa)
    bset = sample(inds, numb)
    if len(set(aset) & set(bset)) == 0:
        bset = bset[:-1] + [choice(aset)]
    A = fill(c, acol, aset)
    B = fill(c, bcol, bset)
    gi = hconcat(A, B)
    res = set(aset) | set(bset)
    go = fill(c, 6, res)
    if choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

IntegerSet = FrozenSet[Integer]

Element = Union[Object, Grid]

TWO = 2

SIX = 6

TWO_BY_TWO = (2, 2)

def equality(
    a: Any,
    b: Any
) -> Boolean:
    """ equality """
    return a == b

def combine(
    a: Container,
    b: Container
) -> Container:
    """ union """
    return type(a)((*a, *b))

def intersection(
    a: FrozenSet,
    b: FrozenSet
) -> FrozenSet:
    """ returns the intersection of two containers """
    return a & b

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

def other(
    container: Container,
    value: Any
) -> Any:
    """ other value in the container """
    return first(remove(value, container))

def branch(
    condition: Boolean,
    if_value: Any,
    else_value: Any
) -> Any:
    """ if else branching """
    return if_value if condition else else_value

def apply(
    function: Callable,
    container: Container
) -> Container:
    """ apply function to each item in container """
    return type(container)(function(e) for e in container)

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

def numcolors(
    element: Element
) -> IntegerSet:
    """ number of colors occurring in object or grid """
    return len(palette(element))

def hsplit(
    grid: Grid,
    n: Integer
) -> Tuple:
    """ split grid horizontally """
    h, w = len(grid), len(grid[0]) // n
    offset = len(grid[0]) % n != 0
    return tuple(crop(grid, (0, w * i + i * offset), (h, w)) for i in range(n))

def vsplit(
    grid: Grid,
    n: Integer
) -> Tuple:
    """ split grid vertically """
    h, w = len(grid) // n, len(grid[0])
    offset = len(grid) % n != 0
    return tuple(crop(grid, (h * i + i * offset, 0), (h, w)) for i in range(n))

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_dae9d2b5(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = hsplit(I, TWO)
    x1 = apply(numcolors, x0)
    x2 = equality(x1, TWO_BY_TWO)
    x3 = branch(x2, hsplit, vsplit)
    x4 = x3(I, TWO)
    x5 = first(x4)
    x6 = last(x4)
    x7 = palette(x5)
    x8 = palette(x6)
    x9 = intersection(x7, x8)
    x10 = first(x9)
    x11 = palette(x5)
    x12 = other(x11, x10)
    x13 = palette(x6)
    x14 = other(x13, x10)
    x15 = shape(x5)
    x16 = canvas(x10, x15)
    x17 = ofcolor(x5, x12)
    x18 = ofcolor(x6, x14)
    x19 = combine(x17, x18)
    x20 = fill(x16, SIX, x19)
    return x20


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("E4", E4_IN, E4_OUT),
        ("E5", E5_IN, E5_OUT),
        ("T", T_IN, T_OUT),
        ("T2", T2_IN, T2_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_dae9d2b5(inp)
        assert pred == _to_grid(expected), f"{name} failed"

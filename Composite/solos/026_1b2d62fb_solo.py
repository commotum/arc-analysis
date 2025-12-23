# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "1b2d62fb"
SERIAL = "026"
URL    = "https://arcprize.org/play?task=1b2d62fb"

# --- Code Golf Concepts ---
CONCEPTS = [
    "detect_wall",
    "separate_images",
    "pattern_intersection",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 9, 9, 1, 9, 9, 9],
    [0, 0, 9, 1, 9, 9, 0],
    [9, 0, 9, 1, 9, 9, 0],
    [0, 0, 0, 1, 9, 0, 0],
    [0, 9, 9, 1, 9, 9, 9],
], dtype=int)

E1_OUT = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 8, 8],
    [0, 0, 0],
], dtype=int)

E2_IN = np.array([
    [0, 0, 0, 1, 9, 0, 0],
    [9, 0, 9, 1, 9, 9, 9],
    [0, 9, 9, 1, 9, 9, 9],
    [0, 0, 0, 1, 9, 9, 9],
    [0, 9, 9, 1, 9, 9, 9],
], dtype=int)

E2_OUT = np.array([
    [0, 8, 8],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
], dtype=int)

E3_IN = np.array([
    [9, 0, 0, 1, 9, 0, 9],
    [9, 0, 0, 1, 0, 9, 0],
    [9, 0, 0, 1, 9, 0, 0],
    [0, 9, 9, 1, 0, 9, 9],
    [0, 0, 9, 1, 0, 9, 0],
], dtype=int)

E3_OUT = np.array([
    [0, 8, 0],
    [0, 0, 8],
    [0, 8, 8],
    [8, 0, 0],
    [8, 0, 0],
], dtype=int)

E4_IN = np.array([
    [0, 9, 9, 1, 9, 0, 9],
    [9, 0, 0, 1, 9, 0, 0],
    [9, 9, 9, 1, 9, 9, 9],
    [0, 9, 0, 1, 0, 0, 0],
    [9, 0, 0, 1, 9, 0, 0],
], dtype=int)

E4_OUT = np.array([
    [0, 0, 0],
    [0, 8, 8],
    [0, 0, 0],
    [8, 0, 8],
    [0, 8, 8],
], dtype=int)

E5_IN = np.array([
    [0, 9, 9, 1, 9, 0, 9],
    [9, 0, 9, 1, 9, 9, 9],
    [9, 9, 9, 1, 0, 0, 9],
    [9, 0, 0, 1, 9, 0, 0],
    [9, 9, 9, 1, 0, 0, 9],
], dtype=int)

E5_OUT = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 8, 8],
    [0, 0, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [9, 9, 0, 1, 0, 9, 0],
    [0, 9, 9, 1, 0, 0, 0],
    [9, 9, 0, 1, 0, 9, 0],
    [9, 9, 9, 1, 9, 0, 9],
    [0, 9, 9, 1, 0, 9, 9],
], dtype=int)

T_OUT = np.array([
    [0, 0, 8],
    [8, 0, 0],
    [0, 0, 8],
    [0, 0, 0],
    [8, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
    return [[8 * (not A | B) for A, B in zip(A, A[4:])] for A in j]


# --- Code Golf Solution (Compressed) ---
def q(g):
    return [eval('8>>r.pop(0)+r[3],' * 3) for r in g]


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

def ofcolor(
    grid: Grid,
    value: Integer
) -> Indices:
    """ indices of all grid cells with value """
    return frozenset((i, j) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value)

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

def generate_1b2d62fb(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(8, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 14))
    bgc = 0
    remcols = remove(bgc, cols)
    barcol = choice(remcols)
    remcols = remove(barcol, remcols)
    cola = choice(remcols)
    colb = choice(remcols)
    canv = canvas(0, (h, w))
    inds = totuple(asindices(canv))
    gbar = canvas(barcol, (h, 1))
    mp = (h * w) // 2
    devrng = (0, mp)
    deva = unifint(diff_lb, diff_ub, devrng)
    devb = unifint(diff_lb, diff_ub, devrng)
    sgna = choice((+1, -1))
    sgnb = choice((+1, -1))
    deva = sgna * deva
    devb = sgnb * devb
    numa = mp + deva
    numb = mp + devb
    numa = max(min(h * w - 1, numa), 1)
    numb = max(min(h * w - 1, numb), 1)
    a = sample(inds, numa)
    b = sample(inds, numb)
    gia = fill(canv, cola, a)
    gib = fill(canv, colb, b)
    gi = hconcat(hconcat(gia, gbar), gib)
    go = fill(canv, 8, ofcolor(gia, 0) & ofcolor(gib, 0))
    if choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Numerical = Union[Integer, IntegerTuple]

IntegerSet = FrozenSet[Integer]

Element = Union[Object, Grid]

ZERO = 0

ONE = 1

EIGHT = 8

def halve(
    n: Numerical
) -> Numerical:
    """ scaling by one half """
    return n // 2 if isinstance(n, int) else (n[0] // 2, n[1] // 2)

def equality(
    a: Any,
    b: Any
) -> Boolean:
    """ equality """
    return a == b

def intersection(
    a: FrozenSet,
    b: FrozenSet
) -> FrozenSet:
    """ returns the intersection of two containers """
    return a & b

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

def astuple(
    a: Integer,
    b: Integer
) -> IntegerTuple:
    """ constructs a tuple """
    return (a, b)

def branch(
    condition: Boolean,
    if_value: Any,
    else_value: Any
) -> Any:
    """ if else branching """
    return if_value if condition else else_value

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

def numcolors(
    element: Element
) -> IntegerSet:
    """ number of colors occurring in object or grid """
    return len(palette(element))

def toobject(
    patch: Patch,
    grid: Grid
) -> Object:
    """ object from patch and grid """
    h, w = len(grid), len(grid[0])
    return frozenset((grid[i][j], (i, j)) for i, j in toindices(patch) if 0 <= i < h and 0 <= j < w)

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

def connect(
    a: IntegerTuple,
    b: IntegerTuple
) -> Indices:
    """ line between two points """
    ai, aj = a
    bi, bj = b
    si = min(ai, bi)
    ei = max(ai, bi) + 1
    sj = min(aj, bj)
    ej = max(aj, bj) + 1
    if ai == bi:
        return frozenset((ai, j) for j in range(sj, ej))
    elif aj == bj:
        return frozenset((i, aj) for i in range(si, ei))
    elif bi - ai == bj - aj:
        return frozenset((i, j) for i, j in zip(range(si, ei), range(sj, ej)))
    elif bi - ai == aj - bj:
        return frozenset((i, j) for i, j in zip(range(si, ei), range(ej - 1, sj - 1, -1)))
    return frozenset()

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

def verify_1b2d62fb(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = width(I)
    x1 = halve(x0)
    x2 = tojvec(x1)
    x3 = height(I)
    x4 = decrement(x3)
    x5 = astuple(x4, x1)
    x6 = connect(x2, x5)
    x7 = toobject(x6, I)
    x8 = numcolors(x7)
    x9 = equality(x8, ONE)
    x10 = branch(x9, lefthalf, tophalf)
    x11 = branch(x9, righthalf, bottomhalf)
    x12 = x10(I)
    x13 = x11(I)
    x14 = shape(x12)
    x15 = canvas(ZERO, x14)
    x16 = ofcolor(x12, ZERO)
    x17 = ofcolor(x13, ZERO)
    x18 = intersection(x16, x17)
    x19 = fill(x15, EIGHT, x18)
    return x19


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
        pred = verify_1b2d62fb(inp)
        assert pred == _to_grid(expected), f"{name} failed"

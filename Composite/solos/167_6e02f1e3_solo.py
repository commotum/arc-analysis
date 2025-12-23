# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "6e02f1e3"
SERIAL = "167"
URL    = "https://arcprize.org/play?task=6e02f1e3"

# --- Code Golf Concepts ---
CONCEPTS = [
    "count_different_colors",
    "associate_images_to_numbers",
]

# --- Example Grids ---
E1_IN = np.array([
    [2, 2, 2],
    [3, 2, 3],
    [3, 3, 3],
], dtype=int)

E1_OUT = np.array([
    [5, 0, 0],
    [0, 5, 0],
    [0, 0, 5],
], dtype=int)

E2_IN = np.array([
    [3, 3, 3],
    [4, 2, 2],
    [4, 4, 2],
], dtype=int)

E2_OUT = np.array([
    [0, 0, 5],
    [0, 5, 0],
    [5, 0, 0],
], dtype=int)

E3_IN = np.array([
    [4, 4, 4],
    [4, 4, 4],
    [4, 4, 4],
], dtype=int)

E3_OUT = np.array([
    [5, 5, 5],
    [0, 0, 0],
    [0, 0, 0],
], dtype=int)

E4_IN = np.array([
    [3, 3, 3],
    [3, 3, 3],
    [3, 3, 3],
], dtype=int)

E4_OUT = np.array([
    [5, 5, 5],
    [0, 0, 0],
    [0, 0, 0],
], dtype=int)

E5_IN = np.array([
    [4, 4, 4],
    [4, 4, 4],
    [3, 3, 3],
], dtype=int)

E5_OUT = np.array([
    [5, 0, 0],
    [0, 5, 0],
    [0, 0, 5],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [4, 4, 4],
    [2, 3, 2],
    [3, 2, 3],
], dtype=int)

T_OUT = np.array([
    [0, 0, 5],
    [0, 5, 0],
    [5, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
    return [[[5, 5, 5], [0, 0, 0], [0, 0, 0]], [[5, 0, 0], [0, 5, 0], [0, 0, 5]], [[0, 0, 5], [0, 5, 0], [5, 0, 0]]][len(set((v for r in j for v in r))) - 1]


# --- Code Golf Solution (Compressed) ---
def q(i):
    return [[5 * (y == x % len({*str(i)}) % 3) for x in b'\x1e\x19\x14'] for y in (0, 1, 2)]


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, sample, shuffle, uniform

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

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

def generate_6e02f1e3(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    d = unifint(diff_lb, diff_ub, (3, 30))
    c = canvas(0, (d, d))
    inds = list(asindices(c))
    shuffle(inds)
    num = d ** 2
    numcols = choice((1, 2, 3))
    chcols = sample(cols, numcols)
    if len(chcols) == 1:
        gi = canvas(chcols[0], (d, d))
        go = canvas(0, (d, d))
        go = fill(go, 5, connect((0, 0), (0, d - 1)))
    elif len(chcols) == 2:
        c1, c2 = chcols
        mp = (d ** 2) // 2
        nc1 = unifint(diff_lb, diff_ub, (1, mp))
        a = inds[:nc1]
        b = inds[nc1:]
        gi = fill(c, c1, a)
        gi = fill(gi, c2, b)
        go = fill(canvas(0, (d, d)), 5, connect((0, 0), (d - 1, d - 1)))
    elif len(chcols) == 3:
        c1, c2, c3 = chcols
        kk = d ** 2
        a = int(1/3 * kk)
        b = int(2/3 * kk)
        adev = unifint(diff_lb, diff_ub, (0, a - 1))
        bdev = unifint(diff_lb, diff_ub, (0, kk - b - 1))
        a -= adev
        b -= bdev
        x1, x2, x3 = inds[:a], inds[a:b], inds[b:]
        gi = fill(c, c1, x1)
        gi = fill(gi, c2, x2)
        gi = fill(gi, c3, x3)
        go = fill(canvas(0, (d, d)), 5, connect((d - 1, 0), (0, d - 1)))
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Numerical = Union[Integer, IntegerTuple]

IntegerSet = FrozenSet[Integer]

Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

ZERO = 0

TWO = 2

THREE = 3

FIVE = 5

ORIGIN = (0, 0)

def equality(
    a: Any,
    b: Any
) -> Boolean:
    """ equality """
    return a == b

def decrement(
    x: Numerical
) -> Numerical:
    """ decrementing """
    return x - 1 if isinstance(x, int) else (x[0] - 1, x[1] - 1)

def toivec(
    i: Integer
) -> IntegerTuple:
    """ vector pointing vertically """
    return (i, 0)

def tojvec(
    j: Integer
) -> IntegerTuple:
    """ vector pointing horizontally """
    return (0, j)

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

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_6e02f1e3(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = numcolors(I)
    x1 = equality(x0, THREE)
    x2 = height(I)
    x3 = decrement(x2)
    x4 = toivec(x3)
    x5 = branch(x1, x4, ORIGIN)
    x6 = equality(x0, TWO)
    x7 = shape(I)
    x8 = decrement(x7)
    x9 = width(I)
    x10 = decrement(x9)
    x11 = tojvec(x10)
    x12 = branch(x6, x8, x11)
    x13 = shape(I)
    x14 = canvas(ZERO, x13)
    x15 = connect(x5, x12)
    x16 = fill(x14, FIVE, x15)
    return x16


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
        pred = verify_6e02f1e3(inp)
        assert pred == _to_grid(expected), f"{name} failed"

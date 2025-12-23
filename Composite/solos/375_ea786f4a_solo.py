# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "ea786f4a"
SERIAL = "375"
URL    = "https://arcprize.org/play?task=ea786f4a"

# --- Code Golf Concepts ---
CONCEPTS = [
    "pattern_modification",
    "draw_line_from_point",
    "diagonals",
]

# --- Example Grids ---
E1_IN = np.array([
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1],
], dtype=int)

E1_OUT = np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0],
], dtype=int)

E2_IN = np.array([
    [2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2],
    [2, 2, 0, 2, 2],
    [2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2],
], dtype=int)

E2_OUT = np.array([
    [0, 2, 2, 2, 0],
    [2, 0, 2, 0, 2],
    [2, 2, 0, 2, 2],
    [2, 0, 2, 0, 2],
    [0, 2, 2, 2, 0],
], dtype=int)

E3_IN = np.array([
    [3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 0, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3],
], dtype=int)

E3_OUT = np.array([
    [0, 3, 3, 3, 3, 3, 0],
    [3, 0, 3, 3, 3, 0, 3],
    [3, 3, 0, 3, 0, 3, 3],
    [3, 3, 3, 0, 3, 3, 3],
    [3, 3, 0, 3, 0, 3, 3],
    [3, 0, 3, 3, 3, 0, 3],
    [0, 3, 3, 3, 3, 3, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    [6, 6, 6, 6, 6, 0, 6, 6, 6, 6, 6],
    [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
], dtype=int)

T_OUT = np.array([
    [0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0],
    [6, 0, 6, 6, 6, 6, 6, 6, 6, 0, 6],
    [6, 6, 0, 6, 6, 6, 6, 6, 0, 6, 6],
    [6, 6, 6, 0, 6, 6, 6, 0, 6, 6, 6],
    [6, 6, 6, 6, 0, 6, 0, 6, 6, 6, 6],
    [6, 6, 6, 6, 6, 0, 6, 6, 6, 6, 6],
    [6, 6, 6, 6, 0, 6, 0, 6, 6, 6, 6],
    [6, 6, 6, 0, 6, 6, 6, 0, 6, 6, 6],
    [6, 6, 0, 6, 6, 6, 6, 6, 0, 6, 6],
    [6, 0, 6, 6, 6, 6, 6, 6, 6, 0, 6],
    [0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
 for A in range(len(j)):j[A][A]=j[-A-1][A]=0
 return j


# --- Code Golf Solution (Compressed) ---
def q(m, i=0):
    for r in m:
        r[i] = r[~i] = 0
        i += 1
    return m


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

def shoot(
    start: IntegerTuple,
    direction: IntegerTuple
) -> Indices:
    """ line from starting point and direction """
    return connect(start, (start[0] + 42 * direction[0], start[1] + 42 * direction[1]))

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

def generate_ea786f4a(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 14))
    w = unifint(diff_lb, diff_ub, (1, 14))
    mp = (h, w)
    h = 2 * h + 1
    w = 2 * w + 1
    linc = choice(cols)
    remcols = remove(linc, cols)
    gi = canvas(linc, (h, w))
    inds = remove(mp, asindices(gi))
    ncols = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(remcols, ncols)
    obj = {(choice(ccols), ij) for ij in inds}
    gi = paint(gi, obj)
    ln1 = shoot(mp, (-1, -1))
    ln2 = shoot(mp, (1, 1))
    ln3 = shoot(mp, (-1, 1))
    ln4 = shoot(mp, (1, -1))
    go = fill(gi, linc, ln1 | ln2 | ln3 | ln4)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Numerical = Union[Integer, IntegerTuple]

Piece = Union[Grid, Patch]

UNITY = (1, 1)

NEG_UNITY = (-1, -1)

UP_RIGHT = (-1, 1)

DOWN_LEFT = (1, -1)

def halve(
    n: Numerical
) -> Numerical:
    """ scaling by one half """
    return n // 2 if isinstance(n, int) else (n[0] // 2, n[1] // 2)

def combine(
    a: Container,
    b: Container
) -> Container:
    """ union """
    return type(a)((*a, *b))

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

def fork(
    outer: Callable,
    a: Callable,
    b: Callable
) -> Callable:
    """ creates a wrapper function """
    return lambda x: outer(a(x), b(x))

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

def index(
    grid: Grid,
    loc: IntegerTuple
) -> Integer:
    """ color at location """
    i, j = loc
    h, w = len(grid), len(grid[0])
    if not (0 <= i < h and 0 <= j < w):
        return None
    return grid[loc[0]][loc[1]]

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_ea786f4a(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = shape(I)
    x1 = halve(x0)
    x2 = rbind(shoot, UP_RIGHT)
    x3 = rbind(shoot, DOWN_LEFT)
    x4 = fork(combine, x2, x3)
    x5 = rbind(shoot, UNITY)
    x6 = rbind(shoot, NEG_UNITY)
    x7 = fork(combine, x5, x6)
    x8 = fork(combine, x4, x7)
    x9 = index(I, x1)
    x10 = x8(x1)
    x11 = fill(I, x9, x10)
    return x11


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_ea786f4a(inp)
        assert pred == _to_grid(expected), f"{name} failed"

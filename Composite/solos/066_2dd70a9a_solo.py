# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "2dd70a9a"
SERIAL = "066"
URL    = "https://arcprize.org/play?task=2dd70a9a"

# --- Code Golf Concepts ---
CONCEPTS = [
    "draw_line_from_point",
    "direction_guessing",
    "maze",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 0, 0, 8, 8, 8, 8, 8, 0, 0, 8, 0, 8, 8, 8, 0, 8, 0, 8],
    [0, 8, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 8, 0, 8, 0, 0, 0, 0],
    [8, 8, 8, 8, 8, 0, 8, 0, 8, 0, 0, 0, 8, 8, 8, 0, 0, 2, 0, 0],
    [8, 0, 8, 8, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 0, 2, 0, 0],
    [8, 0, 0, 8, 8, 0, 0, 0, 0, 0, 0, 8, 0, 8, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0],
    [0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 8, 8],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 8, 0, 0, 0, 0, 0, 8, 8],
    [0, 8, 0, 0, 0, 0, 8, 8, 8, 0, 8, 0, 0, 8, 0, 8, 8, 0, 0, 0],
    [8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 0, 0, 0, 8, 8, 8, 0, 0, 8, 8, 8, 0, 8, 0, 0, 8, 8],
    [0, 0, 0, 0, 0, 0, 8, 8, 0, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 8],
    [0, 0, 0, 3, 0, 0, 0, 8, 0, 8, 0, 8, 0, 0, 8, 0, 0, 8, 0, 8],
    [0, 0, 0, 3, 0, 0, 8, 8, 8, 0, 0, 0, 8, 8, 8, 8, 0, 0, 0, 0],
    [0, 8, 0, 0, 0, 0, 0, 8, 0, 8, 8, 0, 8, 0, 8, 0, 8, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 0, 0, 8, 8, 0],
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 8, 0, 0, 8, 8, 8, 0, 0, 0, 0, 8],
    [0, 0, 0, 0, 8, 8, 8, 8, 0, 0, 8, 0, 0, 0, 0, 8, 8, 8, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [0, 0, 0, 0, 8, 8, 8, 8, 8, 0, 0, 8, 0, 8, 8, 8, 0, 8, 0, 8],
    [0, 8, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 8, 0, 8, 0, 0, 0, 0],
    [8, 8, 8, 8, 8, 0, 8, 0, 8, 0, 0, 0, 8, 8, 8, 0, 0, 2, 0, 0],
    [8, 0, 8, 8, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 0, 2, 0, 0],
    [8, 0, 0, 8, 8, 0, 0, 0, 0, 0, 0, 8, 0, 8, 0, 0, 0, 3, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 3, 0, 0],
    [8, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 8, 0],
    [0, 0, 8, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8],
    [8, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 8, 8],
    [0, 0, 0, 3, 0, 0, 0, 0, 0, 8, 0, 0, 8, 0, 0, 0, 0, 0, 8, 8],
    [0, 8, 0, 3, 0, 0, 8, 8, 8, 0, 8, 0, 0, 8, 0, 8, 8, 0, 0, 0],
    [8, 0, 0, 3, 0, 8, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 3, 0, 0, 8, 8, 8, 0, 0, 8, 8, 8, 0, 8, 0, 0, 8, 8],
    [0, 0, 0, 3, 0, 0, 8, 8, 0, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 8],
    [0, 0, 0, 3, 0, 0, 0, 8, 0, 8, 0, 8, 0, 0, 8, 0, 0, 8, 0, 8],
    [0, 0, 0, 3, 0, 0, 8, 8, 8, 0, 0, 0, 8, 8, 8, 8, 0, 0, 0, 0],
    [0, 8, 0, 0, 0, 0, 0, 8, 0, 8, 8, 0, 8, 0, 8, 0, 8, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 0, 0, 8, 8, 0],
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 8, 0, 0, 8, 8, 8, 0, 0, 0, 0, 8],
    [0, 0, 0, 0, 8, 8, 8, 8, 0, 0, 8, 0, 0, 0, 0, 8, 8, 8, 0, 0],
], dtype=int)

E2_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
    [0, 3, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 3, 0, 0, 0, 0, 0, 8, 0, 0],
    [0, 0, 0, 0, 0, 0, 8, 0, 0, 8],
    [0, 8, 0, 8, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 0, 0, 0],
    [0, 8, 8, 0, 0, 2, 0, 0, 0, 0],
    [0, 0, 8, 0, 0, 2, 0, 0, 0, 0],
    [0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
    [0, 3, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 3, 0, 0, 0, 0, 0, 8, 0, 0],
    [0, 3, 3, 3, 3, 3, 8, 0, 0, 8],
    [0, 8, 0, 8, 0, 3, 0, 0, 0, 0],
    [0, 0, 0, 8, 0, 3, 0, 0, 0, 0],
    [0, 8, 8, 0, 0, 2, 0, 0, 0, 0],
    [0, 0, 8, 0, 0, 2, 0, 0, 0, 0],
    [0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E3_IN = np.array([
    [0, 0, 0, 0, 0, 8, 0, 8, 0, 0, 8, 0, 0, 8, 0],
    [0, 0, 0, 8, 0, 0, 8, 0, 0, 0, 0, 8, 0, 8, 8],
    [8, 0, 0, 0, 8, 8, 8, 0, 0, 0, 0, 8, 8, 8, 0],
    [0, 0, 0, 0, 0, 8, 0, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 8, 0],
    [0, 3, 3, 0, 0, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 8, 0],
    [0, 8, 8, 0, 0, 8, 0, 0, 8, 0, 8, 8, 0, 0, 0],
    [0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0],
    [8, 0, 0, 0, 0, 0, 0, 8, 8, 8, 0, 0, 0, 0, 0],
    [0, 8, 0, 0, 8, 0, 8, 0, 0, 0, 8, 8, 8, 8, 0],
    [0, 0, 0, 0, 0, 8, 0, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 8, 8, 0, 8, 0, 0, 8, 0, 0, 8],
    [0, 8, 0, 0, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [0, 0, 0, 0, 0, 8, 0, 8, 0, 0, 8, 0, 0, 8, 0],
    [0, 0, 0, 8, 0, 0, 8, 0, 0, 0, 0, 8, 0, 8, 8],
    [8, 0, 0, 0, 8, 8, 8, 0, 0, 0, 0, 8, 8, 8, 0],
    [0, 0, 0, 0, 0, 8, 0, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 8, 0],
    [0, 3, 3, 3, 3, 3, 3, 3, 8, 0, 0, 0, 8, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 8, 0, 8, 0],
    [0, 8, 8, 0, 0, 8, 0, 3, 8, 0, 8, 8, 0, 0, 0],
    [0, 8, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
    [8, 2, 2, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 8, 0],
    [8, 0, 0, 0, 0, 0, 0, 8, 8, 8, 0, 0, 0, 0, 0],
    [0, 8, 0, 0, 8, 0, 8, 0, 0, 0, 8, 8, 8, 8, 0],
    [0, 0, 0, 0, 0, 8, 0, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 8, 8, 0, 8, 0, 0, 8, 0, 0, 8],
    [0, 8, 0, 0, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [8, 8, 8, 8, 0, 0, 0, 0, 0, 8, 8, 0, 0],
    [8, 0, 0, 0, 0, 8, 2, 2, 0, 0, 0, 0, 0],
    [0, 8, 0, 0, 8, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 8, 0, 0, 0, 0, 0, 8, 0, 0, 0, 8],
    [0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 0, 0, 8],
    [0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 8, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0],
    [8, 0, 8, 3, 3, 0, 0, 0, 0, 0, 8, 0, 0],
    [0, 8, 8, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0],
    [0, 0, 0, 0, 0, 0, 8, 8, 0, 0, 0, 0, 0],
    [0, 8, 8, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0],
], dtype=int)

T_OUT = np.array([
    [8, 8, 8, 8, 0, 0, 0, 0, 0, 8, 8, 0, 0],
    [8, 0, 0, 0, 0, 8, 2, 2, 3, 3, 0, 0, 0],
    [0, 8, 0, 0, 8, 8, 0, 0, 0, 3, 0, 0, 0],
    [0, 0, 8, 0, 0, 0, 0, 0, 8, 3, 0, 0, 8],
    [0, 0, 8, 0, 0, 0, 8, 0, 0, 3, 0, 0, 8],
    [0, 0, 0, 8, 0, 0, 0, 0, 8, 3, 8, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 8, 0, 0],
    [8, 0, 8, 3, 3, 3, 3, 3, 3, 3, 8, 0, 0],
    [0, 8, 8, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0],
    [0, 0, 0, 0, 0, 0, 8, 8, 0, 0, 0, 0, 0],
    [0, 8, 8, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(*args, **kwargs):
    raise NotImplementedError("Barnacles solution not available for 066")


# --- Code Golf Solution (Compressed) ---
def q(r):
    (n, e), (l, d) = [(n, e) for n in range(len(r)) for e in range(len(r)) if r[e][n] % 2]
    return f([[*n] for n in r], n, e, 2, 1) or f(r, n, e, 2, -1) if l > n else [[*n] for n in zip(*p([[*n] for n in zip(*r)]))]


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, randint, sample, uniform

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

Piece = Union[Grid, Patch]

def identity(
    x: Any
) -> Any:
    """ identity function """
    return x

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

def lrcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of lower right corner """
    return tuple(map(max, zip(*toindices(patch))))

def toindices(
    patch: Patch
) -> Indices:
    """ indices of object cells """
    if len(patch) == 0:
        return frozenset()
    if isinstance(next(iter(patch))[1], tuple):
        return frozenset(index for value, index in patch)
    return patch

def rot90(
    grid: Grid
) -> Grid:
    """ quarter clockwise rotation """
    return tuple(row for row in zip(*grid[::-1]))

def rot180(
    grid: Grid
) -> Grid:
    """ half rotation """
    return tuple(tuple(row[::-1]) for row in grid[::-1])

def rot270(
    grid: Grid
) -> Grid:
    """ quarter anticlockwise rotation """
    return tuple(tuple(row[::-1]) for row in zip(*grid[::-1]))[::-1]

def hmirror(
    piece: Piece
) -> Piece:
    """ mirroring along horizontal """
    if isinstance(piece, tuple):
        return piece[::-1]
    d = ulcorner(piece)[0] + lrcorner(piece)[0]
    if isinstance(next(iter(piece))[1], tuple):
        return frozenset((v, (d - i, j)) for v, (i, j) in piece)
    return frozenset((d - i, j) for i, j in piece)

def vmirror(
    piece: Piece
) -> Piece:
    """ mirroring along vertical """
    if isinstance(piece, tuple):
        return tuple(row[::-1] for row in piece)
    d = ulcorner(piece)[1] + lrcorner(piece)[1]
    if isinstance(next(iter(piece))[1], tuple):
        return frozenset((v, (i, d - j)) for v, (i, j) in piece)
    return frozenset((i, d - j) for i, j in piece)

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

def cmirror(
    piece: Piece
) -> Piece:
    """ mirroring along counterdiagonal """
    if isinstance(piece, tuple):
        return tuple(zip(*(r[::-1] for r in piece[::-1])))
    return vmirror(dmirror(vmirror(piece)))

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

def generate_2dd70a9a(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (2, 3))
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc, fgc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    if choice((True, False)):
        oh = unifint(diff_lb, diff_ub, (5, h - 2))
        ow = unifint(diff_lb, diff_ub, (3, w - 2))
        loci = randint(1, h - oh - 1)
        locj = randint(1, w - ow - 1)
        hli = randint(loci+2, loci+oh-3)
        sp = {(loci+oh-1, locj), (loci+oh-2, locj)}
        ep = {(loci, locj+ow-1), (loci+1, locj+ow-1)}
        bp1 = (hli-1, locj)
        bp2 = (hli, locj+ow)
        ln1 = connect((loci+oh-1, locj), (hli, locj))
        ln2 = connect((hli, locj), (hli, locj+ow-1))
        ln3 = connect((hli, locj+ow-1), (loci+2, locj+ow-1))
    else:
        oh = unifint(diff_lb, diff_ub, (3, h-2))
        ow = unifint(diff_lb, diff_ub, (3, w-2))
        loci = randint(1, h - oh - 1)
        locj = randint(1, w - ow - 1)
        if choice((True, False)):
            sp1j = randint(locj, locj+ow-3)
            ep1j = locj
        else:
            ep1j = randint(locj, locj+ow-3)
            sp1j = locj
        sp = {(loci, sp1j), (loci, sp1j+1)}
        ep = {(loci+oh-1, ep1j), (loci+oh-1, ep1j+1)}
        bp1 = (loci, locj+ow)
        bp2 = (loci+oh, locj+ow-1)
        ln1 = connect((loci, sp1j+2), (loci, locj+ow-1))
        ln2 = connect((loci, locj+ow-1), (loci+oh-1, locj+ow-1))
        ln3 = connect((loci+oh-1, ep1j+2), (loci+oh-1, locj+ow-1))
    gi = fill(gi, 3, sp)
    gi = fill(gi, 2, ep)
    go = fill(go, 3, sp)
    go = fill(go, 2, ep)
    lns = ln1 | ln2 | ln3
    bps = {bp1, bp2}
    gi = fill(gi, fgc, bps)
    go = fill(go, fgc, bps)
    go = fill(go, 3, lns)
    inds = ofcolor(go, bgc)
    namt = unifint(diff_lb, diff_ub, (0, len(inds) // 2))
    noise = sample(totuple(inds), namt)
    gi = fill(gi, fgc, noise)
    go = fill(go, fgc, noise)
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Numerical = Union[Integer, IntegerTuple]

IntegerSet = FrozenSet[Integer]

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

ContainerContainer = Container[Container]

ONE = 1

TWO = 2

THREE = 3

F = False

T = True

DOWN = (1, 0)

RIGHT = (0, 1)

UP = (-1, 0)

LEFT = (0, -1)

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

def merge(
    containers: ContainerContainer
) -> Container:
    """ merging """
    return type(containers)(e for c in containers for e in c)

def both(
    a: Boolean,
    b: Boolean
) -> Boolean:
    """ logical and """
    return a and b

def either(
    a: Boolean,
    b: Boolean
) -> Boolean:
    """ logical or """
    return a or b

def sfilter(
    container: Container,
    condition: Callable
) -> Container:
    """ keep elements in container that satisfy condition """
    return type(container)(e for e in container if condition(e))

def mfilter(
    container: Container,
    function: Callable
) -> FrozenSet:
    """ filter and merge """
    return merge(sfilter(container, function))

def extract(
    container: Container,
    condition: Callable
) -> Any:
    """ first element of container that satisfies condition """
    return next(e for e in container if condition(e))

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

def product(
    a: Container,
    b: Container
) -> FrozenSet:
    """ cartesian product """
    return frozenset((i, j) for j in b for i in a)

def branch(
    condition: Boolean,
    if_value: Any,
    else_value: Any
) -> Any:
    """ if else branching """
    return if_value if condition else else_value

def compose(
    outer: Callable,
    inner: Callable
) -> Callable:
    """ function composition """
    return lambda x: outer(inner(x))

def chain(
    h: Callable,
    g: Callable,
    f: Callable
) -> Callable:
    """ function composition with three functions """
    return lambda x: h(g(f(x)))

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

def lbind(
    function: Callable,
    fixed: Any
) -> Callable:
    """ fix the leftmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda y: function(fixed, y)
    elif n == 3:
        return lambda y, z: function(fixed, y, z)
    else:
        return lambda y, z, a: function(fixed, y, z, a)

def fork(
    outer: Callable,
    a: Callable,
    b: Callable
) -> Callable:
    """ creates a wrapper function """
    return lambda x: outer(a(x), b(x))

def apply(
    function: Callable,
    container: Container
) -> Container:
    """ apply function to each item in container """
    return type(container)(function(e) for e in container)

def mapply(
    function: Callable,
    container: ContainerContainer
) -> FrozenSet:
    """ apply and merge """
    return merge(apply(function, container))

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

def colorfilter(
    objs: Objects,
    value: Integer
) -> Objects:
    """ filter objects by color """
    return frozenset(obj for obj in objs if next(iter(obj))[0] == value)

def asindices(
    grid: Grid
) -> Indices:
    """ indices of all grid cells """
    return frozenset((i, j) for i in range(len(grid)) for j in range(len(grid[0])))

def urcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of upper right corner """
    return tuple(map(lambda ix: {0: min, 1: max}[ix[0]](ix[1]), enumerate(zip(*toindices(patch)))))

def llcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of lower left corner """
    return tuple(map(lambda ix: {0: max, 1: min}[ix[0]](ix[1]), enumerate(zip(*toindices(patch)))))

def dneighbors(
    loc: IntegerTuple
) -> Indices:
    """ directly adjacent indices """
    return frozenset({(loc[0] - 1, loc[1]), (loc[0] + 1, loc[1]), (loc[0], loc[1] - 1), (loc[0], loc[1] + 1)})

def ineighbors(
    loc: IntegerTuple
) -> Indices:
    """ diagonally adjacent indices """
    return frozenset({(loc[0] - 1, loc[1] - 1), (loc[0] - 1, loc[1] + 1), (loc[0] + 1, loc[1] - 1), (loc[0] + 1, loc[1] + 1)})

def neighbors(
    loc: IntegerTuple
) -> Indices:
    """ adjacent indices """
    return dneighbors(loc) | ineighbors(loc)

def objects(
    grid: Grid,
    univalued: Boolean,
    diagonal: Boolean,
    without_bg: Boolean
) -> Objects:
    """ objects occurring on the grid """
    bg = mostcolor(grid) if without_bg else None
    objs = set()
    occupied = set()
    h, w = len(grid), len(grid[0])
    unvisited = asindices(grid)
    diagfun = neighbors if diagonal else dneighbors
    for loc in unvisited:
        if loc in occupied:
            continue
        val = grid[loc[0]][loc[1]]
        if val == bg:
            continue
        obj = {(val, loc)}
        cands = {loc}
        while len(cands) > 0:
            neighborhood = set()
            for cand in cands:
                v = grid[cand[0]][cand[1]]
                if (val == v) if univalued else (v != bg):
                    obj.add((v, cand))
                    occupied.add(cand)
                    neighborhood |= {
                        (i, j) for i, j in diagfun(cand) if 0 <= i < h and 0 <= j < w
                    }
            cands = neighborhood - occupied
        objs.add(frozenset(obj))
    return frozenset(objs)

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

def vline(
    patch: Patch
) -> Boolean:
    """ whether the piece forms a vertical line """
    return height(patch) == len(patch) and width(patch) == 1

def hline(
    patch: Patch
) -> Boolean:
    """ whether the piece forms a horizontal line """
    return width(patch) == len(patch) and height(patch) == 1

def manhattan(
    a: Patch,
    b: Patch
) -> Integer:
    """ closest manhattan distance between two patches """
    return min(abs(ai - bi) + abs(aj - bj) for ai, aj in toindices(a) for bi, bj in toindices(b))

def adjacent(
    a: Patch,
    b: Patch
) -> Boolean:
    """ whether two patches are adjacent """
    return manhattan(a, b) == 1

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

def underfill(
    grid: Grid,
    value: Integer,
    patch: Patch
) -> Grid:
    """ fill value at indices that are background """
    h, w = len(grid), len(grid[0])
    bg = mostcolor(grid)
    grid_filled = list(list(row) for row in grid)
    for i, j in toindices(patch):
        if 0 <= i < h and 0 <= j < w:
            if grid_filled[i][j] == bg:
                grid_filled[i][j] = value
    return tuple(tuple(row) for row in grid_filled)

def replace(
    grid: Grid,
    replacee: Integer,
    replacer: Integer
) -> Grid:
    """ color substitution """
    return tuple(tuple(replacer if v == replacee else v for v in r) for r in grid)

def center(
    patch: Patch
) -> IntegerTuple:
    """ center of the patch """
    return (uppermost(patch) + height(patch) // 2, leftmost(patch) + width(patch) // 2)

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

def corners(
    patch: Patch
) -> Indices:
    """ indices of corners """
    return frozenset({ulcorner(patch), urcorner(patch), llcorner(patch), lrcorner(patch)})

def hfrontier(
    location: IntegerTuple
) -> Indices:
    """ horizontal frontier """
    return frozenset((location[0], j) for j in range(30))

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_2dd70a9a(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = ofcolor(I, TWO)
    x1 = vline(x0)
    x2 = branch(x1, dmirror, identity)
    x3 = x2(I)
    x4 = ofcolor(x3, THREE)
    x5 = ofcolor(x3, TWO)
    x6 = center(x4)
    x7 = hfrontier(x6)
    x8 = center(x5)
    x9 = hfrontier(x8)
    x10 = mostcolor(I)
    x11 = palette(I)
    x12 = remove(THREE, x11)
    x13 = remove(TWO, x12)
    x14 = other(x13, x10)
    x15 = replace(x3, THREE, x10)
    x16 = difference(x7, x4)
    x17 = underfill(x15, THREE, x16)
    x18 = replace(x3, TWO, x10)
    x19 = difference(x9, x5)
    x20 = underfill(x18, TWO, x19)
    x21 = objects(x17, T, F, F)
    x22 = colorfilter(x21, THREE)
    x23 = rbind(adjacent, x4)
    x24 = sfilter(x22, x23)
    x25 = objects(x20, T, F, F)
    x26 = colorfilter(x25, TWO)
    x27 = rbind(adjacent, x5)
    x28 = sfilter(x26, x27)
    x29 = mapply(toindices, x24)
    x30 = rbind(equality, x14)
    x31 = lbind(index, x3)
    x32 = compose(x30, x31)
    x33 = rbind(add, LEFT)
    x34 = compose(x32, x33)
    x35 = rbind(add, RIGHT)
    x36 = compose(x32, x35)
    x37 = fork(either, x34, x36)
    x38 = rbind(add, UP)
    x39 = compose(x32, x38)
    x40 = rbind(add, DOWN)
    x41 = compose(x32, x40)
    x42 = fork(either, x39, x41)
    x43 = sfilter(x29, x37)
    x44 = mapply(toindices, x28)
    x45 = sfilter(x44, x42)
    x46 = fork(connect, first, last)
    x47 = product(x43, x45)
    x48 = compose(vline, x46)
    x49 = rbind(toobject, x3)
    x50 = chain(numcolors, x49, x46)
    x51 = matcher(x50, ONE)
    x52 = fork(both, x48, x51)
    x53 = extract(x47, x52)
    x54 = x46(x53)
    x55 = center(x4)
    x56 = center(x5)
    x57 = fork(either, hline, vline)
    x58 = lbind(connect, x55)
    x59 = corners(x54)
    x60 = apply(x58, x59)
    x61 = mfilter(x60, x57)
    x62 = lbind(connect, x56)
    x63 = corners(x54)
    x64 = apply(x62, x63)
    x65 = mfilter(x64, x57)
    x66 = combine(x61, x65)
    x67 = combine(x54, x66)
    x68 = fill(x3, THREE, x67)
    x69 = fill(x68, TWO, x5)
    x70 = x2(x69)
    return x70


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_2dd70a9a(inp)
        assert pred == _to_grid(expected), f"{name} failed"

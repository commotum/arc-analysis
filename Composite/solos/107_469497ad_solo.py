# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "469497ad"
SERIAL = "107"
URL    = "https://arcprize.org/play?task=469497ad"

# --- Code Golf Concepts ---
CONCEPTS = [
    "image_resizing",
    "draw_line_from_point",
    "diagonals",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 0, 0, 3],
    [0, 8, 8, 0, 3],
    [0, 8, 8, 0, 3],
    [0, 0, 0, 0, 3],
    [3, 3, 3, 3, 3],
], dtype=int)

E1_OUT = np.array([
    [2, 0, 0, 0, 0, 0, 0, 2, 3, 3],
    [0, 2, 0, 0, 0, 0, 2, 0, 3, 3],
    [0, 0, 8, 8, 8, 8, 0, 0, 3, 3],
    [0, 0, 8, 8, 8, 8, 0, 0, 3, 3],
    [0, 0, 8, 8, 8, 8, 0, 0, 3, 3],
    [0, 0, 8, 8, 8, 8, 0, 0, 3, 3],
    [0, 2, 0, 0, 0, 0, 2, 0, 3, 3],
    [2, 0, 0, 0, 0, 0, 0, 2, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
], dtype=int)

E2_IN = np.array([
    [0, 0, 0, 0, 7],
    [4, 4, 0, 0, 7],
    [4, 4, 0, 0, 6],
    [0, 0, 0, 0, 6],
    [7, 7, 6, 6, 6],
], dtype=int)

E2_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 7, 7, 7],
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 7, 7, 7],
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 7, 7, 7],
    [4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 7, 7, 7],
    [4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 7, 7, 7],
    [4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 7, 7, 7],
    [4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 6, 6, 6],
    [4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 6, 6, 6],
    [4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 6, 6, 6],
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 6, 6, 6],
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 6, 6, 6],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 6, 6, 6],
    [7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    [7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    [7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6],
], dtype=int)

E3_IN = np.array([
    [0, 0, 0, 0, 9],
    [0, 1, 1, 0, 9],
    [0, 1, 1, 0, 3],
    [0, 0, 0, 0, 3],
    [9, 9, 3, 3, 4],
], dtype=int)

E3_OUT = np.array([
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 9, 9, 9, 9],
    [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 9, 9, 9, 9],
    [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 9, 9, 9, 9],
    [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 9, 9, 9, 9],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 9, 9, 9, 9],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 9, 9, 9, 9],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 9, 9, 9, 9],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 9, 9, 9, 9],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 3, 3, 3, 3],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 3, 3, 3, 3],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 3, 3, 3, 3],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 3, 3, 3, 3],
    [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 3, 3, 3],
    [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 3, 3, 3, 3],
    [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 3, 3, 3, 3],
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 3, 3, 3],
    [9, 9, 9, 9, 9, 9, 9, 9, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4],
    [9, 9, 9, 9, 9, 9, 9, 9, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4],
    [9, 9, 9, 9, 9, 9, 9, 9, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4],
    [9, 9, 9, 9, 9, 9, 9, 9, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 6, 6, 0, 8],
    [0, 6, 6, 0, 8],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 0, 7],
    [8, 8, 1, 7, 9],
], dtype=int)

T_OUT = np.array([
    [0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8],
    [0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8],
    [0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8],
    [0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8],
    [0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8],
    [0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8],
    [0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8],
    [0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8],
    [0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8],
    [0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8],
    [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 1, 1, 1, 1],
    [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 1, 1, 1, 1],
    [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 1, 1, 1, 1],
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 7, 7, 7],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 7, 7, 7],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 7, 7, 7],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 7, 7, 7],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 7, 7, 7],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1, 1, 1, 1, 1, 7, 7, 7, 7, 7, 9, 9, 9, 9, 9],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1, 1, 1, 1, 1, 7, 7, 7, 7, 7, 9, 9, 9, 9, 9],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1, 1, 1, 1, 1, 7, 7, 7, 7, 7, 9, 9, 9, 9, 9],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1, 1, 1, 1, 1, 7, 7, 7, 7, 7, 9, 9, 9, 9, 9],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1, 1, 1, 1, 1, 7, 7, 7, 7, 7, 9, 9, 9, 9, 9],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j,u=range):
 A=len(j);c=len(j[0]);E=len({*sum(j,[])}-{0})
 j=[[j[W//E][l//E]for l in u(c*E)]for W in u(A*E)];A*=E;c*=E
 for k in u(min(A,c),0,-1):
  for W in u(A-k+1):
   for l in u(c-k+1):
    J=j[W][l]
    if J and all(r[l:l+k]==[J]*k for r in j[W:W+k]):
     for a,C in(-1,-1),(-1,k),(k,-1),(k,k):
      e=W+a;K=l+C
      while-1<e<A and-1<K<c and not j[e][K]:j[e][K]=2;e+=a>0 or-1;K+=C>0 or-1
     return j


# --- Code Golf Solution (Compressed) ---
def q(i):
    z = len({*str(i)}) - 5
    r = range(5 * z)
    return [[i[x // z][y // z] or (x - z * 0 ** i[0][1] in [(u := (y - z * 0 ** i[1][0])), z * 2 + ~u]) * 2 for y in r] for x in r]


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

Element = Union[Object, Grid]

def identity(
    x: Any
) -> Any:
    """ identity function """
    return x

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

def backdrop(
    patch: Patch
) -> Indices:
    """ indices in bounding box of patch """
    if len(patch) == 0:
        return frozenset({})
    indices = toindices(patch)
    si, sj = ulcorner(indices)
    ei, ej = lrcorner(patch)
    return frozenset((i, j) for i in range(si, ei + 1) for j in range(sj, ej + 1))

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

def generate_469497ad(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(2, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (3, 6))
    w = unifint(diff_lb, diff_ub, (3, 6))
    bgc, sqc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    sqh = randint(1, h - 2)
    sqw = randint(1, w - 2)
    sqloci = randint(0, h - sqh - 2)
    sqlocj = randint(0, w - sqw - 2)
    sq = backdrop(frozenset({(sqloci, sqlocj), (sqloci + sqh - 1, sqlocj + sqw - 1)}))
    gi = fill(gi, sqc, sq)
    numcub = min(min(min(h, w)+1, 30//(max(h, w))), 7)
    numc = unifint(diff_lb, diff_ub, (2, numcub))
    numaccc = numc - 1
    remcols = remove(bgc, remove(sqc, cols))
    ccols = sample(remcols, numaccc)
    gi = rot180(gi)
    locs = sample(interval(1, min(h, w), 1), numaccc - 1)
    locs = [0] + sorted(locs)
    for c, l in zip(ccols, locs):
        gi = fill(gi, c, shoot((0, l), (0, 1)))
        gi = fill(gi, c, shoot((l, 0), (1, 0)))
    gi = rot180(gi)
    go = upscale(gi, numc)
    rect = ofcolor(go, sqc)
    l1 = shoot(lrcorner(rect), (1, 1))
    l2 = shoot(ulcorner(rect), (-1, -1))
    l3 = shoot(urcorner(rect), (-1, 1))
    l4 = shoot(llcorner(rect), (1, -1))
    ll = l1 | l2 | l3 | l4
    go = fill(go, 2, ll & ofcolor(go, bgc))
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Numerical = Union[Integer, IntegerTuple]

IntegerSet = FrozenSet[Integer]

Piece = Union[Grid, Patch]

ONE = 1

TWO = 2

UNITY = (1, 1)

NEG_UNITY = (-1, -1)

UP_RIGHT = (-1, 1)

DOWN_LEFT = (1, -1)

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

def argmin(
    container: Container,
    compfunc: Callable
) -> Any:
    """ smallest item by custom order """
    return min(container, key=compfunc, default=None)

def decrement(
    x: Numerical
) -> Numerical:
    """ decrementing """
    return x - 1 if isinstance(x, int) else (x[0] - 1, x[1] - 1)

def sfilter(
    container: Container,
    condition: Callable
) -> Container:
    """ keep elements in container that satisfy condition """
    return type(container)(e for e in container if condition(e))

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

def outbox(
    patch: Patch
) -> Indices:
    """ outbox for patch """
    ai, aj = uppermost(patch) - 1, leftmost(patch) - 1
    bi, bj = lowermost(patch) + 1, rightmost(patch) + 1
    si, sj = min(ai, bi), min(aj, bj)
    ei, ej = max(ai, bi), max(aj, bj)
    vlines = {(i, sj) for i in range(si, ei + 1)} | {(i, ej) for i in range(si, ei + 1)}
    hlines = {(si, j) for j in range(sj, ej + 1)} | {(ei, j) for j in range(sj, ej + 1)}
    return frozenset(vlines | hlines)

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_469497ad(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = numcolors(I)
    x1 = decrement(x0)
    x2 = upscale(I, x1)
    x3 = rbind(toobject, I)
    x4 = lbind(ofcolor, I)
    x5 = compose(outbox, x4)
    x6 = chain(numcolors, x3, x5)
    x7 = matcher(x6, ONE)
    x8 = palette(I)
    x9 = sfilter(x8, x7)
    x10 = fork(multiply, height, width)
    x11 = lbind(ofcolor, I)
    x12 = compose(x10, x11)
    x13 = argmin(x9, x12)
    x14 = ofcolor(x2, x13)
    x15 = outbox(x14)
    x16 = toobject(x15, x2)
    x17 = mostcolor(x16)
    x18 = ulcorner(x14)
    x19 = shoot(x18, NEG_UNITY)
    x20 = lrcorner(x14)
    x21 = shoot(x20, UNITY)
    x22 = urcorner(x14)
    x23 = shoot(x22, UP_RIGHT)
    x24 = llcorner(x14)
    x25 = shoot(x24, DOWN_LEFT)
    x26 = combine(x19, x21)
    x27 = combine(x23, x25)
    x28 = combine(x26, x27)
    x29 = ofcolor(x2, x17)
    x30 = intersection(x28, x29)
    x31 = fill(x2, TWO, x30)
    return x31


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_469497ad(inp)
        assert pred == _to_grid(expected), f"{name} failed"

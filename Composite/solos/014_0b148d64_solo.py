# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "0b148d64"
SERIAL = "014"
URL    = "https://arcprize.org/play?task=0b148d64"

# --- Code Golf Concepts ---
CONCEPTS = [
    "detect_grid",
    "separate_images",
    "find_the_intruder",
    "crop",
]

# --- Example Grids ---
E1_IN = np.array([
    [8, 8, 8, 8, 8, 0, 8, 8, 8, 8, 0, 0, 0, 0, 8, 8, 8, 8, 0, 8, 8],
    [8, 0, 0, 8, 0, 8, 0, 8, 8, 8, 0, 0, 0, 0, 8, 8, 8, 0, 0, 0, 8],
    [8, 8, 8, 0, 0, 0, 8, 8, 8, 8, 0, 0, 0, 0, 8, 8, 0, 8, 8, 8, 8],
    [8, 8, 0, 8, 8, 8, 8, 0, 8, 8, 0, 0, 0, 0, 8, 8, 0, 0, 0, 8, 8],
    [8, 8, 8, 8, 0, 8, 8, 0, 8, 8, 0, 0, 0, 0, 8, 8, 8, 0, 8, 8, 8],
    [0, 0, 0, 8, 8, 0, 8, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0],
    [8, 8, 8, 8, 0, 0, 8, 0, 8, 0, 0, 0, 0, 0, 8, 8, 8, 0, 8, 8, 8],
    [8, 0, 0, 8, 0, 0, 8, 8, 0, 8, 0, 0, 0, 0, 8, 0, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 0, 8, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 2, 2, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 8, 8, 0, 8, 8, 0, 8],
    [2, 0, 2, 2, 2, 0, 0, 2, 2, 2, 0, 0, 0, 0, 8, 8, 8, 8, 0, 8, 0],
    [0, 2, 2, 2, 2, 2, 2, 0, 2, 0, 0, 0, 0, 0, 8, 8, 8, 0, 0, 0, 8],
    [2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 0, 0, 0, 0, 8, 8, 0, 8, 8, 8, 0],
    [2, 2, 2, 2, 2, 2, 0, 2, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 0, 0],
    [2, 2, 2, 2, 2, 0, 2, 0, 2, 2, 0, 0, 0, 0, 8, 0, 8, 0, 8, 8, 8],
    [2, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 0, 8, 0, 0, 8],
    [0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0, 0, 8, 0, 0, 0, 8, 8, 0],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 8, 8, 0, 0, 8, 8],
    [2, 0, 2, 2, 0, 2, 2, 2, 2, 2, 0, 0, 0, 0, 8, 8, 8, 0, 8, 8, 8],
], dtype=int)

E1_OUT = np.array([
    [0, 2, 2, 2, 0, 0, 2, 2, 2, 2],
    [2, 0, 2, 2, 2, 0, 0, 2, 2, 2],
    [0, 2, 2, 2, 2, 2, 2, 0, 2, 0],
    [2, 2, 2, 2, 0, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 0, 2, 0, 0],
    [2, 2, 2, 2, 2, 0, 2, 0, 2, 2],
    [2, 2, 0, 2, 2, 0, 0, 0, 0, 0],
    [0, 2, 2, 0, 0, 2, 2, 0, 0, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 0, 2, 2, 0, 2, 2, 2, 2, 2],
], dtype=int)

E2_IN = np.array([
    [2, 0, 2, 2, 2, 2, 0, 0, 0, 0, 2, 0, 2, 2, 2, 2, 0, 0, 2],
    [2, 2, 2, 2, 0, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 0],
    [0, 0, 2, 2, 0, 2, 0, 0, 0, 0, 2, 2, 2, 0, 2, 2, 2, 2, 2],
    [2, 0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
    [0, 2, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 2, 2, 0, 2, 2, 2],
    [2, 2, 2, 0, 2, 0, 2, 0, 0, 0, 2, 0, 2, 2, 2, 2, 0, 2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 3, 3, 3, 3, 3, 0, 3, 3],
    [0, 2, 2, 0, 0, 2, 2, 0, 0, 0, 3, 3, 3, 0, 0, 0, 3, 3, 0],
    [0, 2, 2, 0, 0, 2, 0, 0, 0, 0, 3, 3, 3, 0, 3, 0, 3, 0, 0],
    [2, 2, 2, 0, 0, 2, 2, 0, 0, 0, 3, 3, 0, 0, 0, 3, 3, 3, 3],
    [2, 0, 0, 2, 2, 2, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 3, 0, 3],
    [2, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 3, 3, 0, 3, 3, 3, 0, 3],
    [0, 2, 2, 0, 2, 2, 0, 0, 0, 0, 0, 3, 3, 0, 0, 3, 0, 3, 0],
], dtype=int)

E2_OUT = np.array([
    [0, 3, 3, 3, 3, 3, 0, 3, 3],
    [3, 3, 3, 0, 0, 0, 3, 3, 0],
    [3, 3, 3, 0, 3, 0, 3, 0, 0],
    [3, 3, 0, 0, 0, 3, 3, 3, 3],
    [3, 0, 0, 0, 3, 0, 3, 0, 3],
    [0, 3, 3, 0, 3, 3, 3, 0, 3],
    [0, 3, 3, 0, 0, 3, 0, 3, 0],
], dtype=int)

E3_IN = np.array([
    [0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1],
    [1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1],
    [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1],
    [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 0, 0, 4, 0, 4, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1],
    [4, 4, 4, 4, 0, 4, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0],
    [4, 0, 4, 0, 0, 4, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1],
    [0, 4, 4, 4, 4, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1],
    [4, 4, 4, 0, 4, 4, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 4, 4, 4, 4, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
    [0, 4, 4, 4, 0, 4, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0],
    [0, 4, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
    [4, 4, 0, 4, 0, 4, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0],
], dtype=int)

E3_OUT = np.array([
    [4, 0, 0, 4, 0, 4],
    [4, 4, 4, 4, 0, 4],
    [4, 0, 4, 0, 0, 4],
    [0, 4, 4, 4, 4, 0],
    [4, 4, 4, 0, 4, 4],
    [0, 4, 4, 4, 4, 0],
    [0, 4, 4, 4, 0, 4],
    [0, 4, 0, 0, 0, 0],
    [4, 4, 0, 4, 0, 4],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [1, 1, 1, 1, 0, 1, 0, 0, 3, 0, 3, 3, 3, 3, 3, 3, 0],
    [1, 0, 1, 0, 1, 1, 0, 0, 0, 3, 0, 3, 3, 3, 0, 0, 0],
    [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 3, 3, 0, 3, 3, 0, 3, 0, 0],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 3, 0, 3, 3, 3, 0, 3, 3],
    [1, 1, 1, 1, 1, 1, 0, 0, 3, 3, 0, 0, 0, 3, 0, 0, 3],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 0, 0, 0, 0, 3, 0, 0, 3, 3, 3, 0, 3, 0, 3, 0, 3],
    [0, 3, 3, 0, 0, 3, 0, 0, 0, 3, 0, 3, 3, 3, 0, 0, 0],
    [3, 3, 3, 3, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3],
    [3, 0, 3, 0, 3, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 0, 3],
    [0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 0, 3, 3, 0],
], dtype=int)

T_OUT = np.array([
    [1, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 1, 1],
    [1, 1, 0, 1, 1, 0],
    [0, 0, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
from collections import*
def p(j):
 A=[x for k in j for x in k];c=Counter(A).most_common(3);c=[c for c in c if c[0]>0][-1][0];j=[k for k in j if c in k];E=[]
 for k in j:
  for W in range(len(k)):
   if k[W]==c:E+=[W]
 return[k[min(E):max(E)+1]for k in j]


# --- Code Golf Solution (Compressed) ---
def q(g):
    return [p(zip(r, *g)) or r[0] for r in [*g] if [*{*r}][2:]]


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

def crop(
    grid: Grid,
    start: IntegerTuple,
    dims: IntegerTuple
) -> Grid:
    """ subgrid specified by start and dimension """
    return tuple(r[start[1]:start[1]+dims[1]] for r in grid[start[0]:start[0]+dims[0]])

def toindices(
    patch: Patch
) -> Indices:
    """ indices of object cells """
    if len(patch) == 0:
        return frozenset()
    if isinstance(next(iter(patch))[1], tuple):
        return frozenset(index for value, index in patch)
    return patch

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

def subgrid(
    patch: Patch,
    grid: Grid
) -> Grid:
    """ smallest subgrid containing object """
    return crop(grid, ulcorner(patch), shape(patch))

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

def generate_0b148d64(diff_lb: float, diff_ub: float) -> dict:
    itv = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (7, 30))
    w = unifint(diff_lb, diff_ub, (7, 30))
    bgc = choice(itv)
    remitv = remove(bgc, itv)
    g = canvas(bgc, (h, w))
    x = randint(3, h - 3)
    y = randint(3, w - 3)
    di = randint(2, h - x - 1)
    dj = randint(2, w - y - 1)
    A = backdrop(frozenset({(0, 0), (x, y)}))
    B = backdrop(frozenset({(x + di, 0), (h - 1, y)}))
    C = backdrop(frozenset({(0, y + dj), (x, w - 1)}))
    D = backdrop(frozenset({(x + di, y + dj), (h - 1, w - 1)}))
    cola = choice(remitv)
    colb = choice(remove(cola, remitv))
    trg = choice((A, B, C, D))
    rem = remove(trg, (A, B, C, D))
    subf = lambda bx: {
        choice(totuple(connect(ulcorner(bx), urcorner(bx)))),
        choice(totuple(connect(ulcorner(bx), llcorner(bx)))),
        choice(totuple(connect(urcorner(bx), lrcorner(bx)))),
        choice(totuple(connect(llcorner(bx), lrcorner(bx)))),
    }
    sampler = lambda bx: set(sample(
        totuple(bx),
        len(bx) - unifint(diff_lb, diff_ub, (0, len(bx) - 1))
    ))
    gi = fill(g, cola, sampler(trg) | subf(trg))
    for r in rem:
        gi = fill(gi, colb, sampler(r) | subf(r))
    go = subgrid(frozenset(trg), gi)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Numerical = Union[Integer, IntegerTuple]

IntegerSet = FrozenSet[Integer]

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

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

def argmin(
    container: Container,
    compfunc: Callable
) -> Any:
    """ smallest item by custom order """
    return min(container, key=compfunc, default=None)

def fork(
    outer: Callable,
    a: Callable,
    b: Callable
) -> Callable:
    """ creates a wrapper function """
    return lambda x: outer(a(x), b(x))

def partition(
    grid: Grid
) -> Objects:
    """ each cell with the same value part of the same object """
    return frozenset(
        frozenset(
            (v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value
        ) for value in palette(grid)
    )

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

def verify_0b148d64(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = partition(I)
    x1 = fork(multiply, height, width)
    x2 = argmin(x0, x1)
    x3 = subgrid(x2, I)
    return x3


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_0b148d64(inp)
        assert pred == _to_grid(expected), f"{name} failed"

# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "8731374e"
SERIAL = "205"
URL    = "https://arcprize.org/play?task=8731374e"

# --- Code Golf Concepts ---
CONCEPTS = [
    "rectangle_guessing",
    "crop",
    "draw_line_from_point",
]

# --- Example Grids ---
E1_IN = np.array([
    [6, 1, 2, 4, 8, 3, 7, 2, 6, 5, 7, 7, 4, 9, 2, 5, 9, 4, 5, 9, 3, 8, 7],
    [6, 0, 1, 0, 4, 8, 6, 1, 1, 2, 1, 2, 6, 6, 6, 5, 8, 7, 4, 1, 7, 5, 6],
    [6, 8, 3, 1, 9, 8, 7, 1, 2, 3, 9, 2, 6, 2, 1, 0, 5, 7, 7, 7, 8, 1, 3],
    [2, 2, 9, 5, 5, 6, 6, 9, 3, 8, 6, 2, 4, 1, 8, 3, 5, 7, 5, 5, 6, 1, 6],
    [1, 7, 6, 4, 7, 0, 1, 7, 9, 1, 7, 6, 9, 6, 6, 8, 4, 6, 8, 8, 9, 8, 0],
    [2, 9, 2, 3, 9, 6, 8, 8, 1, 1, 1, 1, 1, 1, 9, 7, 2, 4, 0, 1, 6, 4, 5],
    [8, 3, 9, 5, 6, 5, 6, 8, 1, 1, 1, 1, 1, 1, 3, 0, 1, 3, 1, 6, 3, 5, 1],
    [0, 7, 2, 6, 5, 2, 0, 7, 1, 1, 1, 1, 2, 1, 2, 2, 3, 0, 7, 5, 1, 8, 8],
    [2, 4, 7, 2, 7, 0, 9, 3, 1, 1, 1, 1, 1, 1, 4, 7, 7, 6, 2, 0, 0, 0, 4],
    [5, 1, 3, 2, 7, 5, 2, 8, 1, 2, 1, 1, 1, 1, 4, 6, 4, 7, 5, 2, 8, 9, 6],
    [6, 8, 2, 6, 8, 4, 6, 7, 1, 1, 1, 1, 1, 1, 8, 2, 1, 7, 9, 1, 2, 9, 1],
    [1, 1, 9, 9, 4, 7, 2, 2, 1, 1, 1, 1, 1, 1, 3, 9, 2, 4, 9, 3, 6, 4, 5],
    [5, 9, 4, 8, 5, 8, 8, 1, 5, 3, 8, 8, 4, 7, 6, 4, 1, 1, 8, 5, 6, 2, 2],
    [1, 1, 4, 7, 9, 1, 5, 6, 8, 2, 3, 2, 2, 4, 4, 8, 6, 5, 6, 8, 5, 8, 3],
    [9, 4, 2, 5, 1, 7, 4, 8, 1, 8, 5, 5, 7, 9, 1, 8, 5, 3, 1, 8, 0, 2, 0],
    [2, 9, 2, 7, 1, 5, 2, 2, 8, 6, 9, 3, 9, 6, 6, 3, 6, 2, 2, 6, 1, 4, 6],
    [6, 5, 3, 7, 0, 9, 1, 3, 2, 6, 5, 0, 6, 1, 0, 5, 2, 7, 1, 4, 8, 4, 1],
], dtype=int)

E1_OUT = np.array([
    [1, 2, 1, 1, 2, 1],
    [1, 2, 1, 1, 2, 1],
    [2, 2, 2, 2, 2, 2],
    [1, 2, 1, 1, 2, 1],
    [2, 2, 2, 2, 2, 2],
    [1, 2, 1, 1, 2, 1],
    [1, 2, 1, 1, 2, 1],
], dtype=int)

E2_IN = np.array([
    [3, 1, 8, 2, 5, 1, 9, 5, 0, 5, 1, 2, 4, 2, 9, 7, 4, 4, 5, 8, 6, 7, 6],
    [5, 6, 8, 3, 9, 8, 4, 1, 2, 1, 5, 3, 2, 4, 6, 1, 8, 7, 6, 6, 9, 9, 0],
    [6, 8, 6, 0, 2, 0, 2, 5, 2, 8, 0, 2, 1, 9, 5, 8, 1, 2, 9, 4, 7, 4, 4],
    [8, 5, 7, 4, 4, 4, 1, 9, 8, 2, 5, 7, 6, 6, 0, 8, 3, 7, 8, 1, 0, 9, 9],
    [0, 3, 8, 2, 6, 4, 9, 5, 3, 5, 4, 9, 5, 5, 4, 0, 8, 1, 5, 2, 1, 1, 0],
    [8, 4, 7, 9, 5, 2, 3, 0, 8, 0, 1, 7, 6, 4, 2, 0, 8, 7, 3, 9, 5, 5, 6],
    [5, 6, 0, 8, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 6, 4, 9, 8, 2, 6, 3, 8, 2],
    [0, 0, 1, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 7, 7, 0, 4, 4, 0, 4, 1, 4],
    [7, 3, 3, 1, 4, 4, 1, 4, 4, 4, 4, 4, 4, 6, 5, 0, 8, 5, 9, 7, 3, 9, 1],
    [9, 3, 0, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 6, 1, 4, 0, 4, 6, 4, 7, 0],
    [5, 0, 8, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 7, 4, 8, 3, 6, 4, 3, 4, 3, 5],
    [4, 6, 4, 3, 4, 4, 4, 4, 4, 4, 1, 4, 4, 2, 6, 1, 0, 8, 1, 1, 8, 8, 1],
    [7, 4, 8, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 2, 1, 5, 7, 9, 2, 5, 0],
    [2, 5, 2, 4, 4, 4, 4, 4, 1, 4, 4, 4, 4, 1, 4, 3, 3, 1, 2, 8, 7, 9, 9],
    [6, 4, 5, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 9, 6, 1, 7, 9, 9, 7, 8],
    [3, 8, 6, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 3, 4, 3, 7, 4, 6, 3, 7, 6],
    [1, 0, 1, 4, 5, 0, 7, 9, 1, 9, 6, 5, 6, 1, 6, 4, 5, 1, 3, 0, 2, 3, 9],
    [4, 6, 4, 6, 2, 7, 0, 8, 5, 9, 4, 1, 7, 0, 9, 1, 3, 7, 7, 5, 4, 1, 7],
    [2, 0, 6, 6, 0, 3, 8, 6, 7, 3, 3, 8, 2, 6, 8, 5, 7, 1, 1, 8, 4, 3, 9],
    [9, 4, 3, 8, 6, 2, 9, 0, 7, 1, 3, 5, 7, 8, 7, 6, 1, 0, 2, 2, 2, 5, 3],
    [3, 8, 2, 2, 3, 6, 2, 4, 0, 2, 3, 9, 9, 1, 6, 3, 4, 6, 7, 9, 7, 0, 8],
    [1, 9, 4, 5, 8, 3, 8, 3, 7, 6, 6, 6, 9, 2, 3, 4, 7, 9, 6, 1, 3, 3, 3],
    [2, 3, 9, 3, 9, 6, 6, 3, 2, 8, 0, 3, 6, 4, 5, 0, 9, 9, 8, 1, 4, 4, 0],
    [7, 6, 6, 4, 1, 9, 6, 8, 5, 3, 2, 5, 6, 8, 9, 6, 4, 2, 6, 3, 4, 7, 9],
    [4, 1, 7, 6, 6, 7, 4, 3, 0, 2, 0, 7, 1, 7, 3, 0, 2, 0, 3, 8, 6, 2, 7],
    [2, 5, 4, 4, 0, 8, 2, 8, 9, 8, 9, 7, 8, 5, 3, 3, 2, 5, 7, 4, 0, 3, 7],
    [2, 5, 5, 0, 0, 4, 2, 4, 9, 9, 3, 1, 6, 1, 1, 6, 5, 9, 8, 3, 7, 4, 2],
], dtype=int)

E2_OUT = np.array([
    [4, 4, 1, 4, 1, 4, 1, 4, 4],
    [4, 4, 1, 4, 1, 4, 1, 4, 4],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [4, 4, 1, 4, 1, 4, 1, 4, 4],
    [4, 4, 1, 4, 1, 4, 1, 4, 4],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [4, 4, 1, 4, 1, 4, 1, 4, 4],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [4, 4, 1, 4, 1, 4, 1, 4, 4],
    [4, 4, 1, 4, 1, 4, 1, 4, 4],
], dtype=int)

E3_IN = np.array([
    [0, 0, 7, 9, 8, 8, 0, 8, 9, 9, 3, 1, 4, 5, 2, 7, 6],
    [6, 0, 9, 2, 7, 2, 8, 4, 3, 3, 2, 7, 7, 5, 9, 4, 0],
    [1, 9, 4, 5, 4, 8, 8, 8, 8, 8, 8, 8, 8, 2, 0, 7, 9],
    [5, 5, 6, 8, 3, 8, 8, 8, 8, 8, 8, 8, 8, 2, 0, 2, 7],
    [8, 2, 3, 2, 9, 8, 8, 8, 8, 8, 8, 8, 8, 0, 7, 6, 4],
    [1, 7, 3, 3, 5, 8, 8, 8, 2, 8, 8, 8, 8, 7, 1, 1, 4],
    [7, 2, 3, 5, 6, 8, 8, 8, 8, 8, 8, 8, 8, 5, 8, 5, 6],
    [5, 2, 7, 3, 5, 8, 8, 8, 8, 8, 8, 8, 8, 1, 4, 4, 6],
    [1, 4, 0, 0, 9, 9, 4, 0, 2, 6, 5, 5, 0, 8, 6, 4, 7],
    [8, 7, 8, 3, 3, 8, 0, 9, 0, 4, 8, 9, 8, 5, 2, 7, 3],
    [2, 0, 2, 8, 2, 0, 8, 4, 4, 3, 2, 6, 8, 7, 4, 7, 2],
    [2, 7, 8, 3, 7, 4, 2, 4, 8, 4, 2, 3, 9, 9, 2, 0, 8],
    [4, 8, 8, 5, 3, 2, 0, 1, 8, 9, 3, 9, 8, 1, 8, 8, 7],
    [3, 9, 9, 9, 1, 6, 1, 9, 4, 7, 5, 5, 3, 2, 9, 3, 0],
    [5, 8, 2, 5, 4, 2, 2, 4, 0, 9, 2, 8, 1, 3, 5, 7, 3],
    [8, 0, 9, 5, 3, 8, 4, 5, 0, 2, 5, 2, 9, 6, 0, 1, 0],
], dtype=int)

E3_OUT = np.array([
    [8, 8, 8, 2, 8, 8, 8, 8],
    [8, 8, 8, 2, 8, 8, 8, 8],
    [8, 8, 8, 2, 8, 8, 8, 8],
    [2, 2, 2, 2, 2, 2, 2, 2],
    [8, 8, 8, 2, 8, 8, 8, 8],
    [8, 8, 8, 2, 8, 8, 8, 8],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [2, 7, 2, 0, 2, 6, 3, 0, 3, 9, 1, 3, 5, 3, 0, 4, 5],
    [4, 4, 8, 7, 0, 7, 9, 1, 4, 9, 5, 2, 0, 8, 5, 3, 2],
    [8, 7, 9, 8, 8, 8, 8, 8, 8, 8, 8, 7, 6, 1, 5, 2, 1],
    [6, 9, 3, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 8, 1, 3, 6],
    [0, 2, 9, 8, 8, 8, 8, 8, 8, 1, 8, 9, 5, 1, 9, 4, 1],
    [5, 2, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8, 3, 6, 7, 9, 5],
    [8, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 1, 7, 3, 7],
    [8, 6, 2, 8, 8, 1, 8, 8, 8, 8, 8, 6, 3, 1, 1, 2, 9],
    [9, 4, 0, 8, 8, 8, 8, 8, 8, 8, 8, 6, 4, 0, 6, 7, 6],
    [6, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 4, 7, 1, 5, 8, 4],
    [4, 0, 3, 8, 8, 8, 8, 8, 8, 8, 8, 4, 2, 4, 3, 4, 5],
    [3, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 4, 8, 7, 7, 1, 8],
    [6, 6, 4, 7, 6, 8, 1, 8, 1, 9, 2, 6, 8, 7, 2, 8, 8],
    [7, 3, 5, 1, 4, 1, 6, 4, 9, 6, 7, 7, 9, 2, 3, 0, 2],
    [9, 2, 2, 5, 4, 8, 3, 9, 9, 9, 5, 9, 6, 1, 4, 6, 9],
    [6, 1, 9, 6, 3, 1, 6, 6, 8, 6, 0, 1, 3, 4, 8, 7, 7],
    [2, 1, 2, 4, 9, 2, 1, 5, 1, 7, 0, 7, 9, 3, 8, 2, 1],
    [7, 1, 9, 4, 2, 8, 4, 3, 6, 2, 8, 0, 8, 5, 3, 5, 9],
    [1, 2, 5, 7, 8, 7, 1, 6, 5, 8, 0, 9, 2, 8, 9, 1, 5],
], dtype=int)

T_OUT = np.array([
    [8, 8, 1, 8, 8, 8, 1, 8],
    [8, 8, 1, 8, 8, 8, 1, 8],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [8, 8, 1, 8, 8, 8, 1, 8],
    [8, 8, 1, 8, 8, 8, 1, 8],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [8, 8, 1, 8, 8, 8, 1, 8],
    [8, 8, 1, 8, 8, 8, 1, 8],
    [8, 8, 1, 8, 8, 8, 1, 8],
    [8, 8, 1, 8, 8, 8, 1, 8],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(*args, **kwargs):
    raise NotImplementedError("Barnacles solution not available for 205")


# --- Code Golf Solution (Compressed) ---
def q(g, k=87):
    return -k * [[min(e + u, key=e.count) for *u, in zip(*g)] for *e, in g] or p([*zip(*g)][sum((u[:6].count(u[0]) > 4 for u in g)) < 6:][::-1], k - 1)


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, randint, sample, shuffle, uniform

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

Piece = Union[Grid, Patch]

ContainerContainer = Container[Container]

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

def asobject(
    grid: Grid
) -> Object:
    """ conversion of grid to object """
    return frozenset((v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r))

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

def vfrontier(
    location: IntegerTuple
) -> Indices:
    """ vertical frontier """
    return frozenset((i, location[1]) for i in range(30))

def hfrontier(
    location: IntegerTuple
) -> Indices:
    """ horizontal frontier """
    return frozenset((location[0], j) for j in range(30))

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

def generate_8731374e(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    inh = randint(5, h - 2)
    inw = randint(5, w - 2)
    bgc, fgc = sample(cols, 2)
    num = unifint(diff_lb, diff_ub, (1, min(inh, inw)))
    mat = canvas(bgc, (inh - 2, inw - 2))
    tol = lambda g: list(list(e) for e in g)
    tot = lambda g: tuple(tuple(e) for e in g)
    mat = fill(mat, fgc, connect((0, 0), (num - 1, num - 1)))
    mat = tol(mat)
    shuffle(mat)
    mat = tol(dmirror(tot(mat)))
    shuffle(mat)
    mat = dmirror(tot(mat))
    sgi = paint(canvas(bgc, (inh, inw)), shift(asobject(mat), (1, 1)))
    inds = ofcolor(sgi, fgc)
    lins = mapply(fork(combine, vfrontier, hfrontier), inds)
    go = fill(sgi, fgc, lins)
    numci = unifint(diff_lb, diff_ub, (3, 10))
    numc = 13 - numci
    ccols = sample(cols, numc)
    c = canvas(-1, (h, w))
    inds = asindices(c)
    obj = {(choice(ccols), ij) for ij in inds}
    gi = paint(c, obj)
    loci = randint(1, h - inh - 1)
    locj = randint(1, w - inw - 1)
    loc = (loci, locj)
    plcd = shift(asobject(sgi), loc)
    gi = paint(gi, plcd)
    a, b = ulcorner(plcd)
    c, d = lrcorner(plcd)
    p1 = choice(totuple(connect((a - 1, b), (a - 1, d))))
    p2 = choice(totuple(connect((a, b - 1), (c, b - 1))))
    p3 = choice(totuple(connect((c + 1, b), (c + 1, d))))
    p4 = choice(totuple(connect((a, d + 1), (c, d + 1))))
    remcols = remove(bgc, ccols)
    fixobj = {
        (choice(remcols), p1), (choice(remcols), p2),
        (choice(remcols), p3), (choice(remcols), p4)
    }
    gi = paint(gi, fixobj)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Numerical = Union[Integer, IntegerTuple]

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

TWO = 2

EIGHT = 8

F = False

T = True

DOWN = (1, 0)

def identity(
    x: Any
) -> Any:
    """ identity function """
    return x

def subtract(
    a: Numerical,
    b: Numerical
) -> Numerical:
    """ subtraction """
    if isinstance(a, int) and isinstance(b, int):
        return a - b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] - b[0], a[1] - b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a - b[0], a - b[1])
    return (a[0] - b, a[1] - b)

def double(
    n: Numerical
) -> Numerical:
    """ scaling by two """
    return n * 2 if isinstance(n, int) else (n[0] * 2, n[1] * 2)

def greater(
    a: Integer,
    b: Integer
) -> Boolean:
    """ greater """
    return a > b

def size(
    container: Container
) -> Integer:
    """ cardinality """
    return len(container)

def argmax(
    container: Container,
    compfunc: Callable
) -> Any:
    """ largest item by custom order """
    return max(container, key=compfunc, default=None)

def initset(
    value: Any
) -> FrozenSet:
    """ initialize container """
    return frozenset({value})

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

def insert(
    value: Any,
    container: FrozenSet
) -> FrozenSet:
    """ insert item into container """
    return container.union(frozenset({value}))

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

def power(
    function: Callable,
    n: Integer
) -> Callable:
    """ power of function """
    if n == 1:
        return function
    return compose(function, power(function, n - 1))

def rapply(
    functions: Container,
    value: Any
) -> Container:
    """ apply each function in container to value """
    return type(functions)(function(value) for function in functions)

def mostcolor(
    element: Element
) -> Integer:
    """ most common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return max(set(values), key=values.count)

def leastcolor(
    element: Element
) -> Integer:
    """ least common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return min(set(values), key=values.count)

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

def crop(
    grid: Grid,
    start: IntegerTuple,
    dims: IntegerTuple
) -> Grid:
    """ subgrid specified by start and dimension """
    return tuple(r[start[1]:start[1]+dims[1]] for r in grid[start[0]:start[0]+dims[0]])

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

def color(
    obj: Object
) -> Integer:
    """ color of object """
    return next(iter(obj))[0]

def rot90(
    grid: Grid
) -> Grid:
    """ quarter clockwise rotation """
    return tuple(row for row in zip(*grid[::-1]))

def subgrid(
    patch: Patch,
    grid: Grid
) -> Grid:
    """ smallest subgrid containing object """
    return crop(grid, ulcorner(patch), shape(patch))

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_8731374e(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = objects(I, T, F, F)
    x1 = argmax(x0, size)
    x2 = color(x1)
    x3 = subgrid(x1, I)
    x4 = lbind(insert, DOWN)
    x5 = compose(lrcorner, asindices)
    x6 = chain(x4, initset, x5)
    x7 = fork(subgrid, x6, identity)
    x8 = matcher(identity, x2)
    x9 = rbind(subtract, TWO)
    x10 = rbind(sfilter, x8)
    x11 = compose(x9, width)
    x12 = chain(size, x10, first)
    x13 = fork(greater, x11, x12)
    x14 = rbind(branch, identity)
    x15 = rbind(x14, x7)
    x16 = chain(initset, x15, x13)
    x17 = fork(rapply, x16, identity)
    x18 = compose(first, x17)
    x19 = compose(x18, rot90)
    x20 = double(EIGHT)
    x21 = double(x20)
    x22 = power(x19, x21)
    x23 = x22(x3)
    x24 = leastcolor(x23)
    x25 = ofcolor(x23, x24)
    x26 = fork(combine, vfrontier, hfrontier)
    x27 = mapply(x26, x25)
    x28 = fill(x23, x24, x27)
    return x28


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_8731374e(inp)
        assert pred == _to_grid(expected), f"{name} failed"

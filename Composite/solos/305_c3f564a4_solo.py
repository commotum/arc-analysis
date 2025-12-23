# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "c3f564a4"
SERIAL = "305"
URL    = "https://arcprize.org/play?task=c3f564a4"

# --- Code Golf Concepts ---
CONCEPTS = [
    "pattern_expansion",
    "image_filling",
]

# --- Example Grids ---
E1_IN = np.array([
    [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1],
    [2, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2],
    [3, 0, 0, 0, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3],
    [4, 0, 0, 0, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4],
    [5, 0, 0, 0, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 0, 0, 5, 1],
    [2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 0, 0, 1, 2],
    [3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3],
    [4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4],
    [5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 0, 0, 0, 0, 4, 5],
    [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 0, 0, 0, 0, 5, 1],
    [2, 3, 4, 5, 1, 2, 0, 0, 0, 1, 0, 0, 0, 0, 1, 2],
    [3, 4, 5, 1, 2, 3, 0, 0, 0, 0, 3, 4, 5, 1, 2, 3],
    [4, 5, 1, 2, 3, 4, 0, 0, 0, 0, 4, 5, 1, 2, 3, 4],
    [5, 1, 2, 3, 4, 5, 0, 0, 0, 0, 5, 1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1],
], dtype=int)

E1_OUT = np.array([
    [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1],
    [2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2],
    [3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3],
    [4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4],
    [5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1],
    [2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2],
    [3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3],
    [4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4],
    [5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1],
    [2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2],
    [3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3],
    [4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4],
    [5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1],
], dtype=int)

E2_IN = np.array([
    [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4],
    [2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5],
    [3, 4, 5, 6, 1, 2, 0, 0, 5, 6, 1, 2, 3, 4, 5, 6],
    [4, 5, 6, 1, 2, 0, 0, 0, 6, 1, 2, 3, 4, 5, 6, 1],
    [5, 6, 1, 2, 3, 0, 0, 0, 1, 2, 3, 4, 5, 6, 1, 2],
    [6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3],
    [1, 2, 3, 4, 5, 6, 1, 2, 0, 0, 0, 6, 1, 2, 3, 4],
    [2, 3, 4, 5, 6, 1, 2, 3, 0, 0, 0, 0, 2, 3, 4, 5],
    [3, 4, 5, 6, 1, 2, 3, 4, 0, 0, 0, 0, 3, 4, 5, 6],
    [0, 0, 0, 0, 2, 3, 4, 5, 0, 0, 0, 0, 4, 5, 6, 1],
    [0, 0, 0, 0, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2],
    [0, 0, 0, 0, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3],
    [0, 0, 0, 0, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4],
    [2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5],
    [3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
    [4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1],
], dtype=int)

E2_OUT = np.array([
    [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4],
    [2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5],
    [3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
    [4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1],
    [5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2],
    [6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3],
    [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4],
    [2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5],
    [3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
    [4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1],
    [5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2],
    [6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3],
    [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4],
    [2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5],
    [3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
    [4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1],
], dtype=int)

E3_IN = np.array([
    [1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2],
    [2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3],
    [3, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4],
    [4, 0, 0, 0, 0, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5],
    [5, 0, 0, 0, 0, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6],
    [6, 0, 0, 0, 0, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7],
    [7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1],
    [1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2],
    [2, 0, 0, 0, 0, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3],
    [3, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4],
    [4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 0, 0, 4, 5],
    [5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 0, 0, 5, 6],
    [6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 0, 0, 0, 0, 7],
    [7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 0, 0, 0, 0, 1],
    [1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 0, 0, 0, 0, 2],
    [2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3],
], dtype=int)

E3_OUT = np.array([
    [1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2],
    [2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3],
    [3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4],
    [4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5],
    [5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6],
    [6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7],
    [7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1],
    [1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2],
    [2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3],
    [3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4],
    [4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5],
    [5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6],
    [6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7],
    [7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1],
    [1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2],
    [2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 3, 4, 5, 6, 7, 8],
    [2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 4, 5, 6, 7, 8, 1],
    [3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2],
    [4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3],
    [5, 6, 0, 0, 0, 0, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4],
    [6, 7, 0, 0, 0, 0, 0, 0, 0, 7, 8, 1, 2, 3, 4, 5],
    [7, 8, 0, 0, 0, 0, 0, 0, 0, 8, 1, 2, 3, 4, 5, 6],
    [8, 1, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7],
    [1, 2, 3, 4, 5, 0, 0, 0, 0, 2, 3, 4, 5, 6, 7, 8],
    [2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1],
    [3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2],
    [4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3],
    [5, 6, 7, 8, 1, 2, 3, 0, 0, 6, 7, 8, 1, 2, 3, 4],
    [6, 7, 8, 1, 2, 3, 4, 0, 0, 7, 8, 1, 2, 3, 4, 5],
    [7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6],
    [8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7],
], dtype=int)

T_OUT = np.array([
    [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8],
    [2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1],
    [3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2],
    [4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3],
    [5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4],
    [6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5],
    [7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6],
    [8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7],
    [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8],
    [2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1],
    [3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2],
    [4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3],
    [5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4],
    [6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5],
    [7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6],
    [8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
	A=len(j);c=[A for c in j for A in c if A]
	if not c:return j
	E=sorted(set(c));k=len(E);W=[[0]*A for c in[0]*A]
	for l in range(A):
		for J in range(A):W[l][J]=E[(l+J)%k]
	return W


# --- Code Golf Solution (Compressed) ---
def q(g):
    return [(9 * [*{*r} - {0}])[g.index(r):][:16] for r in g]


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

def merge(
    containers: ContainerContainer
) -> Container:
    """ merging """
    return type(containers)(e for c in containers for e in c)

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

def normalize(
    patch: Patch
) -> Patch:
    """ moves upper left corner to origin """
    if len(patch) == 0:
        return patch
    return shift(patch, (-uppermost(patch), -leftmost(patch)))

def uppermost(
    patch: Patch
) -> Integer:
    """ row index of uppermost occupied cell """
    return min(i for i, j in toindices(patch))

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

def rot90(
    grid: Grid
) -> Grid:
    """ quarter clockwise rotation """
    return tuple(row for row in zip(*grid[::-1]))

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

def occurrences(
    grid: Grid,
    obj: Object
) -> Indices:
    """ locations of occurrences of object in grid """
    occurrences = set()
    normed = normalize(obj)
    h, w = len(grid), len(grid[0])
    for i in range(h):
        for j in range(w):
            occurs = True
            for v, (a, b) in shift(normed, (i, j)):
                if 0 <= a < h and 0 <= b < w:
                    if grid[a][b] != v:
                        occurs = False
                        break
                else:
                    occurs = False
                    break
            if occurs:
                occurrences.add((i, j))
    return frozenset(occurrences)

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

def generate_c3f564a4(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (7, 30))
    w = unifint(diff_lb, diff_ub, (7, 30))
    p = unifint(diff_lb, diff_ub, (2, min(9, min(h//3, w//3))))
    fixc = choice(cols)
    remcols = remove(fixc, cols)
    ccols = list(sample(remcols, p))
    shuffle(ccols)
    c = canvas(-1, (h, w))
    baseobj = {(cc, (0, jj)) for cc, jj in zip(ccols, range(p))}
    obj = {c for c in baseobj}
    while rightmost(obj) < 2 * max(w, h):
        obj = obj | shift(obj, (0, p))
    if choice((True, False)):
        obj = mapply(lbind(shift, obj), {(jj, 0) for jj in interval(0, h, 1)})
    else:
        obj = mapply(lbind(shift, obj), {(jj, -jj) for jj in interval(0, h, 1)})
    go = paint(c, obj)
    gi = tuple(e for e in go)
    nsq = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 25)))
    maxtr = 4 * nsq
    tr = 0
    succ = 0
    while succ < nsq and tr < maxtr:
        oh = unifint(diff_lb, diff_ub, (2, 5))
        ow = unifint(diff_lb, diff_ub, (2, 5))
        loci = randint(0, h - oh)
        locj = randint(0, w - ow)
        tmpg = fill(gi, fixc, backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)})))
        if len(occurrences(tmpg, baseobj)) > 1 and len([r for r in tmpg if fixc not in r]) > 0 and len([r for r in dmirror(tmpg) if fixc not in r]) > 0:
            gi = tmpg
            succ += 1
        tr += 1
    if choice((True, False)):
        gi = rot90(gi)
        go = rot90(go)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Numerical = Union[Integer, IntegerTuple]

IntegerSet = FrozenSet[Integer]

Element = Union[Object, Grid]

ONE = 1

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

def divide(
    a: Numerical,
    b: Numerical
) -> Numerical:
    """ floor division """
    if isinstance(a, int) and isinstance(b, int):
        return a // b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] // b[0], a[1] // b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a // b[0], a // b[1])
    return (a[0] // b, a[1] // b)

def invert(
    n: Numerical
) -> Numerical:
    """ inversion with respect to addition """
    return -n if isinstance(n, int) else (-n[0], -n[1])

def flip(
    b: Boolean
) -> Boolean:
    """ logical not """
    return not b

def contained(
    value: Any,
    container: Container
) -> Boolean:
    """ element of """
    return value in container

def combine(
    a: Container,
    b: Container
) -> Container:
    """ union """
    return type(a)((*a, *b))

def minimum(
    container: IntegerSet
) -> Integer:
    """ minimum """
    return min(container, default=0)

def argmin(
    container: Container,
    compfunc: Callable
) -> Any:
    """ smallest item by custom order """
    return min(container, key=compfunc, default=None)

def increment(
    x: Numerical
) -> Numerical:
    """ incrementing """
    return x + 1 if isinstance(x, int) else (x[0] + 1, x[1] + 1)

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

def astuple(
    a: Integer,
    b: Integer
) -> IntegerTuple:
    """ constructs a tuple """
    return (a, b)

def product(
    a: Container,
    b: Container
) -> FrozenSet:
    """ cartesian product """
    return frozenset((i, j) for j in b for i in a)

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

def crop(
    grid: Grid,
    start: IntegerTuple,
    dims: IntegerTuple
) -> Grid:
    """ subgrid specified by start and dimension """
    return tuple(r[start[1]:start[1]+dims[1]] for r in grid[start[0]:start[0]+dims[0]])

def lowermost(
    patch: Patch
) -> Integer:
    """ row index of lowermost occupied cell """
    return max(i for i, j in toindices(patch))

def palette(
    element: Element
) -> IntegerSet:
    """ colors occurring in object or grid """
    if isinstance(element, tuple):
        return frozenset({v for r in element for v in r})
    return frozenset({v for v, _ in element})

def asobject(
    grid: Grid
) -> Object:
    """ conversion of grid to object """
    return frozenset((v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r))

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

def hperiod(
    obj: Object
) -> Integer:
    """ horizontal periodicity """
    normalized = normalize(obj)
    w = width(normalized)
    for p in range(1, w):
        offsetted = shift(normalized, (0, -p))
        pruned = frozenset({(c, (i, j)) for c, (i, j) in offsetted if j >= 0})
        if pruned.issubset(normalized):
            return p
    return w

def vperiod(
    obj: Object
) -> Integer:
    """ vertical periodicity """
    normalized = normalize(obj)
    h = height(normalized)
    for p in range(1, h):
        offsetted = shift(normalized, (-p, 0))
        pruned = frozenset({(c, (i, j)) for c, (i, j) in offsetted if i >= 0})
        if pruned.issubset(normalized):
            return p
    return h

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_c3f564a4(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = height(I)
    x1 = vsplit(I, x0)
    x2 = apply(asobject, x1)
    x3 = apply(hperiod, x2)
    x4 = minimum(x3)
    x5 = width(I)
    x6 = hsplit(I, x5)
    x7 = apply(asobject, x6)
    x8 = apply(vperiod, x7)
    x9 = minimum(x8)
    x10 = matcher(hperiod, x4)
    x11 = sfilter(x2, x10)
    x12 = mapply(palette, x11)
    x13 = matcher(vperiod, x9)
    x14 = sfilter(x7, x13)
    x15 = mapply(palette, x14)
    x16 = palette(I)
    x17 = combine(x12, x15)
    x18 = rbind(contained, x17)
    x19 = argmin(x16, x18)
    x20 = asobject(I)
    x21 = matcher(first, x19)
    x22 = compose(flip, x21)
    x23 = sfilter(x20, x22)
    x24 = height(I)
    x25 = divide(x24, x9)
    x26 = increment(x25)
    x27 = width(I)
    x28 = divide(x27, x4)
    x29 = increment(x28)
    x30 = invert(x26)
    x31 = interval(x30, x26, ONE)
    x32 = invert(x29)
    x33 = interval(x32, x29, ONE)
    x34 = product(x31, x33)
    x35 = astuple(x9, x4)
    x36 = lbind(multiply, x35)
    x37 = apply(x36, x34)
    x38 = lbind(shift, x23)
    x39 = mapply(x38, x37)
    x40 = paint(I, x39)
    return x40


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_c3f564a4(inp)
        assert pred == _to_grid(expected), f"{name} failed"

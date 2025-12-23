# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "b2862040"
SERIAL = "279"
URL    = "https://arcprize.org/play?task=b2862040"

# --- Code Golf Concepts ---
CONCEPTS = [
    "recoloring",
    "detect_closed_curves",
    "associate_colors_to_bools",
]

# --- Example Grids ---
E1_IN = np.array([
    [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
    [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
    [9, 1, 1, 1, 9, 9, 9, 1, 9, 9, 9],
    [9, 1, 9, 1, 9, 9, 9, 1, 9, 9, 9],
    [9, 1, 9, 1, 9, 9, 1, 1, 1, 1, 9],
    [9, 1, 1, 1, 9, 9, 9, 1, 9, 9, 9],
    [9, 9, 9, 9, 9, 9, 9, 1, 9, 9, 9],
    [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
    [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
], dtype=int)

E1_OUT = np.array([
    [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
    [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
    [9, 8, 8, 8, 9, 9, 9, 1, 9, 9, 9],
    [9, 8, 9, 8, 9, 9, 9, 1, 9, 9, 9],
    [9, 8, 9, 8, 9, 9, 1, 1, 1, 1, 9],
    [9, 8, 8, 8, 9, 9, 9, 1, 9, 9, 9],
    [9, 9, 9, 9, 9, 9, 9, 1, 9, 9, 9],
    [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
    [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
], dtype=int)

E2_IN = np.array([
    [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
    [9, 1, 1, 1, 1, 1, 9, 9, 1, 9, 9],
    [9, 1, 9, 9, 9, 1, 9, 9, 1, 9, 1],
    [9, 1, 1, 1, 1, 1, 9, 9, 1, 1, 1],
    [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
    [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
    [9, 9, 9, 1, 9, 9, 9, 9, 9, 9, 9],
    [9, 9, 1, 1, 1, 1, 1, 9, 9, 9, 9],
    [9, 9, 9, 1, 9, 1, 9, 9, 9, 9, 9],
    [9, 9, 9, 1, 1, 1, 9, 9, 1, 1, 1],
    [9, 9, 9, 9, 9, 9, 9, 9, 1, 9, 1],
    [1, 1, 9, 9, 9, 9, 9, 9, 1, 1, 1],
], dtype=int)

E2_OUT = np.array([
    [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
    [9, 8, 8, 8, 8, 8, 9, 9, 1, 9, 9],
    [9, 8, 9, 9, 9, 8, 9, 9, 1, 9, 1],
    [9, 8, 8, 8, 8, 8, 9, 9, 1, 1, 1],
    [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
    [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
    [9, 9, 9, 8, 9, 9, 9, 9, 9, 9, 9],
    [9, 9, 8, 8, 8, 8, 8, 9, 9, 9, 9],
    [9, 9, 9, 8, 9, 8, 9, 9, 9, 9, 9],
    [9, 9, 9, 8, 8, 8, 9, 9, 8, 8, 8],
    [9, 9, 9, 9, 9, 9, 9, 9, 8, 9, 8],
    [1, 1, 9, 9, 9, 9, 9, 9, 8, 8, 8],
], dtype=int)

E3_IN = np.array([
    [9, 9, 9, 9, 9, 1, 9, 9, 9, 9, 9, 9, 9],
    [9, 9, 9, 9, 9, 9, 9, 9, 1, 9, 9, 9, 9],
    [9, 9, 1, 9, 9, 9, 9, 1, 1, 1, 1, 9, 9],
    [9, 1, 1, 1, 1, 9, 9, 9, 1, 9, 9, 9, 9],
    [9, 1, 9, 9, 1, 9, 9, 9, 1, 9, 9, 9, 9],
    [9, 1, 1, 1, 1, 9, 9, 9, 1, 1, 1, 9, 9],
    [9, 9, 9, 9, 1, 9, 9, 9, 9, 9, 9, 9, 9],
    [9, 9, 9, 9, 1, 9, 9, 9, 9, 9, 9, 9, 9],
    [9, 1, 9, 9, 9, 9, 9, 1, 1, 1, 9, 9, 9],
    [1, 1, 1, 9, 9, 9, 9, 9, 9, 1, 9, 9, 9],
    [9, 1, 9, 9, 9, 9, 1, 9, 1, 1, 9, 9, 9],
    [1, 1, 9, 9, 9, 9, 1, 1, 1, 9, 9, 9, 9],
], dtype=int)

E3_OUT = np.array([
    [9, 9, 9, 9, 9, 1, 9, 9, 9, 9, 9, 9, 9],
    [9, 9, 9, 9, 9, 9, 9, 9, 1, 9, 9, 9, 9],
    [9, 9, 8, 9, 9, 9, 9, 1, 1, 1, 1, 9, 9],
    [9, 8, 8, 8, 8, 9, 9, 9, 1, 9, 9, 9, 9],
    [9, 8, 9, 9, 8, 9, 9, 9, 1, 9, 9, 9, 9],
    [9, 8, 8, 8, 8, 9, 9, 9, 1, 1, 1, 9, 9],
    [9, 9, 9, 9, 8, 9, 9, 9, 9, 9, 9, 9, 9],
    [9, 9, 9, 9, 8, 9, 9, 9, 9, 9, 9, 9, 9],
    [9, 1, 9, 9, 9, 9, 9, 1, 1, 1, 9, 9, 9],
    [1, 1, 1, 9, 9, 9, 9, 9, 9, 1, 9, 9, 9],
    [9, 1, 9, 9, 9, 9, 1, 9, 1, 1, 9, 9, 9],
    [1, 1, 9, 9, 9, 9, 1, 1, 1, 9, 9, 9, 9],
], dtype=int)

E4_IN = np.array([
    [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
    [9, 1, 1, 1, 1, 1, 1, 9, 9, 9, 9, 1, 1, 1, 1],
    [9, 9, 1, 9, 9, 9, 1, 9, 9, 9, 9, 1, 9, 9, 1],
    [9, 9, 1, 1, 1, 9, 1, 9, 9, 9, 1, 1, 1, 9, 1],
    [9, 9, 9, 9, 1, 1, 1, 9, 9, 9, 9, 9, 9, 9, 1],
    [9, 9, 9, 9, 1, 9, 9, 9, 1, 1, 1, 9, 9, 9, 9],
    [9, 9, 9, 9, 9, 9, 9, 9, 1, 9, 1, 1, 9, 9, 9],
    [9, 9, 9, 9, 9, 9, 9, 9, 1, 1, 1, 9, 9, 9, 9],
    [1, 1, 1, 1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
    [1, 9, 9, 1, 9, 9, 9, 1, 9, 1, 9, 9, 9, 9, 9],
    [1, 1, 1, 1, 9, 9, 9, 1, 1, 1, 1, 1, 9, 9, 9],
    [1, 9, 9, 9, 9, 9, 9, 9, 9, 1, 9, 9, 9, 9, 9],
    [9, 9, 9, 9, 9, 1, 9, 9, 9, 9, 9, 9, 9, 9, 9],
    [9, 9, 9, 9, 1, 1, 9, 9, 9, 9, 9, 9, 1, 1, 9],
], dtype=int)

E4_OUT = np.array([
    [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
    [9, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 1, 1, 1, 1],
    [9, 9, 8, 9, 9, 9, 8, 9, 9, 9, 9, 1, 9, 9, 1],
    [9, 9, 8, 8, 8, 9, 8, 9, 9, 9, 1, 1, 1, 9, 1],
    [9, 9, 9, 9, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 1],
    [9, 9, 9, 9, 8, 9, 9, 9, 8, 8, 8, 9, 9, 9, 9],
    [9, 9, 9, 9, 9, 9, 9, 9, 8, 9, 8, 8, 9, 9, 9],
    [9, 9, 9, 9, 9, 9, 9, 9, 8, 8, 8, 9, 9, 9, 9],
    [8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
    [8, 9, 9, 8, 9, 9, 9, 1, 9, 1, 9, 9, 9, 9, 9],
    [8, 8, 8, 8, 9, 9, 9, 1, 1, 1, 1, 1, 9, 9, 9],
    [8, 9, 9, 9, 9, 9, 9, 9, 9, 1, 9, 9, 9, 9, 9],
    [9, 9, 9, 9, 9, 1, 9, 9, 9, 9, 9, 9, 9, 9, 9],
    [9, 9, 9, 9, 1, 1, 9, 9, 9, 9, 9, 9, 1, 1, 9],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [1, 1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
    [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 1, 9, 9, 9],
    [9, 9, 9, 1, 1, 1, 1, 1, 9, 9, 9, 1, 9, 9, 9],
    [9, 9, 9, 9, 1, 9, 9, 1, 9, 9, 9, 1, 9, 9, 9],
    [9, 9, 9, 9, 1, 9, 9, 1, 9, 9, 9, 1, 9, 9, 9],
    [9, 9, 9, 9, 1, 1, 1, 1, 9, 9, 9, 1, 9, 9, 1],
    [9, 9, 9, 9, 9, 9, 9, 1, 9, 9, 9, 1, 1, 1, 1],
    [1, 1, 1, 1, 9, 9, 9, 1, 9, 9, 9, 1, 9, 9, 1],
    [1, 9, 9, 1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 1],
    [1, 9, 9, 1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 1, 1],
    [1, 1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
    [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
    [9, 9, 9, 1, 1, 1, 1, 1, 1, 9, 9, 9, 1, 1, 9],
    [9, 9, 9, 1, 9, 9, 9, 9, 1, 9, 9, 9, 9, 1, 9],
    [9, 9, 9, 1, 9, 9, 9, 9, 1, 9, 9, 9, 9, 1, 9],
    [9, 9, 9, 1, 1, 1, 1, 1, 1, 1, 9, 9, 9, 1, 9],
], dtype=int)

T_OUT = np.array([
    [1, 1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
    [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 1, 9, 9, 9],
    [9, 9, 9, 8, 8, 8, 8, 8, 9, 9, 9, 1, 9, 9, 9],
    [9, 9, 9, 9, 8, 9, 9, 8, 9, 9, 9, 1, 9, 9, 9],
    [9, 9, 9, 9, 8, 9, 9, 8, 9, 9, 9, 1, 9, 9, 9],
    [9, 9, 9, 9, 8, 8, 8, 8, 9, 9, 9, 1, 9, 9, 1],
    [9, 9, 9, 9, 9, 9, 9, 8, 9, 9, 9, 1, 1, 1, 1],
    [1, 1, 1, 1, 9, 9, 9, 8, 9, 9, 9, 1, 9, 9, 1],
    [1, 9, 9, 1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 1],
    [1, 9, 9, 1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 1, 1],
    [1, 1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
    [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
    [9, 9, 9, 8, 8, 8, 8, 8, 8, 9, 9, 9, 1, 1, 9],
    [9, 9, 9, 8, 9, 9, 9, 9, 8, 9, 9, 9, 9, 1, 9],
    [9, 9, 9, 8, 9, 9, 9, 9, 8, 9, 9, 9, 9, 1, 9],
    [9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 1, 9],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(*args, **kwargs):
    raise NotImplementedError("Barnacles solution not available for 279")


# --- Code Golf Solution (Compressed) ---
def q(g, i=94):
    return g * ~i or p([[9 & r.pop() % [q + 9, 9 | 3 - q][i < 9] or (i < 0) * 9 for q in [0] + r[:0:-1]] for *r, in zip(*g)], i - 1)


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, randint, uniform

Boolean = bool

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Objects = FrozenSet[Object]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

ContainerContainer = Container[Container]

F = False

T = True

def flip(
    b: Boolean
) -> Boolean:
    """ logical not """
    return not b

def difference(
    a: Container,
    b: Container
) -> Container:
    """ difference """
    return type(a)(e for e in a if e not in b)

def merge(
    containers: ContainerContainer
) -> Container:
    """ merging """
    return type(containers)(e for c in containers for e in c)

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

def compose(
    outer: Callable,
    inner: Callable
) -> Callable:
    """ function composition """
    return lambda x: outer(inner(x))

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

def shape(
    piece: Piece
) -> IntegerTuple:
    """ height and width of grid or patch """
    return (height(piece), width(piece))

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

def normalize(
    patch: Patch
) -> Patch:
    """ moves upper left corner to origin """
    if len(patch) == 0:
        return patch
    return shift(patch, (-uppermost(patch), -leftmost(patch)))

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

def bordering(
    patch: Patch,
    grid: Grid
) -> Boolean:
    """ whether a patch is adjacent to a grid border """
    return uppermost(patch) == 0 or leftmost(patch) == 0 or lowermost(patch) == len(grid) - 1 or rightmost(patch) == len(grid[0]) - 1

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

def corners(
    patch: Patch
) -> Indices:
    """ indices of corners """
    return frozenset({ulcorner(patch), urcorner(patch), llcorner(patch), lrcorner(patch)})

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

def box(
    patch: Patch
) -> Indices:
    """ outline of patch """
    if len(patch) == 0:
        return patch
    ai, aj = ulcorner(patch)
    bi, bj = lrcorner(patch)
    si, sj = min(ai, bi), min(aj, bj)
    ei, ej = max(ai, bi), max(aj, bj)
    vlines = {(i, sj) for i in range(si, ei + 1)} | {(i, ej) for i in range(si, ei + 1)}
    hlines = {(si, j) for j in range(sj, ej + 1)} | {(ei, j) for j in range(sj, ej + 1)}
    return frozenset(vlines | hlines)

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

def generate_b2862040(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (8,))
    while True:
        h = unifint(diff_lb, diff_ub, (10, 30))
        w = unifint(diff_lb, diff_ub, (10, 30))
        nobjs = unifint(diff_lb, diff_ub, (1, (h * w) // 16))
        succ = 0
        tr = 0
        maxtr = 10 * nobjs
        bgc = choice(cols)
        remcols = remove(bgc, cols)
        gi = canvas(bgc, (h, w))
        inds = asindices(gi)
        while succ < nobjs and tr < maxtr:
            tr += 1
            oh = randint(3, 6)
            ow = randint(3, 6)
            obj = box(frozenset({(0, 0), (oh - 1, ow - 1)}))
            if choice((True, False)):
                nkeep = unifint(diff_lb, diff_ub, (2, len(obj) - 1))
                nrem = len(obj) - nkeep
                obj = remove(choice(totuple(obj - corners(obj))), obj)
                for k in range(nrem - 1):
                    xx = sfilter(obj, lambda ij: len(dneighbors(ij) & obj) == 1)
                    if len(xx) == 0:
                        break
                    obj = remove(choice(totuple(xx)), obj)
            npert = unifint(diff_lb, diff_ub, (0, oh + ow))
            objcands = outbox(obj) | outbox(outbox(obj)) | outbox(outbox(outbox(obj)))
            obj = set(obj)
            for k in range(npert):
                obj.add(choice(totuple((objcands - obj) & (mapply(dneighbors, obj) & objcands))))
            obj = normalize(obj)
            oh, ow = shape(obj)
            cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
            if len(cands) == 0:
                continue
            loc = choice(totuple(cands))
            plcd = shift(obj, loc)
            if plcd.issubset(inds):
                gi = fill(gi, choice(remcols), plcd)
                succ += 1
                inds = (inds - plcd) - mapply(neighbors, plcd)
        objs = objects(gi, T, F, F)
        bobjs = colorfilter(objs, bgc)
        objsm = mfilter(bobjs, compose(flip, rbind(bordering, gi)))
        if len(objsm) > 0:
            res = mfilter(objs - bobjs, rbind(adjacent, objsm))
            go = fill(gi, 8, res)
            break
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
EIGHT = 8

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_b2862040(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = objects(I, T, F, F)
    x1 = mostcolor(I)
    x2 = colorfilter(x0, x1)
    x3 = rbind(bordering, I)
    x4 = compose(flip, x3)
    x5 = mfilter(x2, x4)
    x6 = difference(x0, x2)
    x7 = apply(toindices, x6)
    x8 = rbind(adjacent, x5)
    x9 = mfilter(x7, x8)
    x10 = fill(I, EIGHT, x9)
    return x10


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("E4", E4_IN, E4_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_b2862040(inp)
        assert pred == _to_grid(expected), f"{name} failed"

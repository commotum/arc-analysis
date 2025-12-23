# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "d07ae81c"
SERIAL = "324"
URL    = "https://arcprize.org/play?task=d07ae81c"

# --- Code Golf Concepts ---
CONCEPTS = [
    "draw_line_from_point",
    "diagonals",
    "color_guessing",
]

# --- Example Grids ---
E1_IN = np.array([
    [8, 8, 8, 2, 2, 2, 2, 8, 8, 8, 8, 8],
    [8, 8, 8, 2, 2, 2, 2, 8, 8, 8, 8, 8],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [8, 8, 8, 2, 4, 2, 2, 8, 8, 8, 8, 8],
    [8, 8, 8, 2, 2, 2, 2, 8, 8, 8, 8, 8],
    [8, 8, 8, 2, 2, 2, 2, 8, 8, 8, 8, 8],
    [8, 1, 8, 2, 2, 2, 2, 8, 8, 8, 8, 8],
    [8, 8, 8, 2, 2, 2, 2, 8, 8, 8, 8, 8],
    [8, 8, 8, 2, 2, 2, 2, 8, 8, 8, 8, 8],
    [8, 8, 8, 2, 2, 2, 2, 8, 8, 8, 8, 8],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [8, 8, 8, 2, 2, 2, 2, 8, 8, 8, 8, 8],
    [8, 8, 8, 2, 2, 2, 2, 8, 8, 8, 8, 8],
    [8, 8, 8, 2, 2, 2, 2, 8, 8, 8, 8, 8],
], dtype=int)

E1_OUT = np.array([
    [8, 8, 8, 2, 2, 2, 2, 8, 8, 1, 8, 8],
    [1, 8, 8, 2, 2, 2, 2, 8, 1, 8, 8, 8],
    [2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2],
    [2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2],
    [2, 2, 2, 4, 2, 4, 2, 2, 2, 2, 2, 2],
    [8, 8, 8, 2, 4, 2, 2, 8, 8, 8, 8, 8],
    [8, 8, 8, 4, 2, 4, 2, 8, 8, 8, 8, 8],
    [1, 8, 1, 2, 2, 2, 4, 8, 8, 8, 8, 8],
    [8, 1, 8, 2, 2, 2, 2, 1, 8, 8, 8, 8],
    [1, 8, 1, 2, 2, 2, 2, 8, 1, 8, 8, 8],
    [8, 8, 8, 4, 2, 2, 2, 8, 8, 1, 8, 8],
    [8, 8, 8, 2, 4, 2, 2, 8, 8, 8, 1, 8],
    [2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 4],
    [2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2],
    [8, 8, 8, 2, 2, 2, 2, 8, 8, 1, 8, 8],
    [8, 8, 8, 2, 2, 2, 2, 8, 8, 8, 1, 8],
    [8, 8, 8, 2, 2, 2, 2, 8, 8, 8, 8, 1],
], dtype=int)

E2_IN = np.array([
    [3, 3, 3, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3],
    [3, 3, 3, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3],
    [3, 3, 3, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3],
    [3, 3, 3, 1, 1, 1, 2, 1, 1, 3, 3, 3, 3, 3],
    [3, 3, 3, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [3, 3, 3, 1, 1, 1, 1, 1, 1, 3, 3, 8, 3, 3],
    [3, 3, 3, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3],
    [3, 3, 3, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3],
    [3, 3, 3, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3],
], dtype=int)

E2_OUT = np.array([
    [3, 3, 3, 2, 1, 1, 1, 1, 1, 8, 3, 3, 3, 3],
    [3, 3, 3, 1, 2, 1, 1, 1, 2, 3, 3, 3, 3, 3],
    [3, 3, 3, 1, 1, 2, 1, 2, 1, 3, 3, 3, 3, 3],
    [3, 3, 3, 1, 1, 1, 2, 1, 1, 3, 3, 3, 3, 3],
    [3, 3, 3, 1, 1, 2, 1, 2, 1, 3, 3, 3, 3, 3],
    [1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1],
    [1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2],
    [1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1],
    [3, 8, 3, 1, 1, 1, 1, 1, 1, 3, 3, 8, 3, 3],
    [8, 3, 3, 1, 1, 1, 1, 1, 1, 3, 8, 3, 8, 3],
    [3, 3, 3, 1, 1, 1, 1, 1, 1, 8, 3, 3, 3, 8],
    [3, 3, 3, 1, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3],
], dtype=int)

E3_IN = np.array([
    [1, 1, 6, 6, 6, 6, 1, 1, 1, 1, 6, 6, 6, 6, 6],
    [1, 1, 6, 6, 6, 6, 1, 1, 1, 1, 6, 6, 6, 6, 6],
    [1, 1, 6, 6, 6, 6, 1, 1, 1, 1, 6, 6, 6, 6, 6],
    [1, 1, 6, 6, 6, 6, 1, 1, 1, 1, 6, 6, 6, 6, 6],
    [1, 1, 6, 6, 6, 6, 1, 1, 1, 1, 6, 6, 6, 6, 6],
    [1, 1, 6, 6, 6, 6, 1, 8, 1, 1, 6, 6, 6, 6, 6],
    [1, 1, 6, 6, 6, 6, 1, 1, 1, 1, 6, 6, 6, 6, 6],
    [1, 1, 6, 6, 6, 6, 1, 1, 1, 1, 6, 6, 6, 6, 6],
    [1, 1, 6, 6, 3, 6, 1, 1, 1, 1, 6, 6, 6, 6, 6],
    [1, 1, 6, 6, 6, 6, 1, 1, 1, 1, 6, 6, 6, 6, 6],
    [1, 1, 6, 6, 6, 6, 1, 1, 1, 1, 6, 6, 3, 6, 6],
    [1, 1, 6, 6, 6, 6, 1, 1, 1, 1, 6, 6, 6, 6, 6],
    [1, 1, 6, 6, 6, 6, 1, 1, 1, 1, 6, 6, 6, 6, 6],
    [1, 1, 6, 6, 6, 6, 1, 1, 1, 1, 6, 6, 6, 6, 6],
    [1, 1, 6, 6, 6, 6, 1, 1, 1, 1, 6, 6, 6, 6, 6],
], dtype=int)

E3_OUT = np.array([
    [1, 1, 3, 6, 6, 6, 1, 1, 1, 1, 6, 6, 3, 6, 6],
    [1, 1, 6, 3, 6, 6, 1, 1, 1, 1, 6, 3, 6, 6, 6],
    [1, 1, 6, 6, 3, 6, 1, 1, 1, 1, 3, 6, 6, 6, 6],
    [1, 1, 6, 6, 6, 3, 1, 1, 1, 8, 6, 6, 6, 6, 6],
    [8, 1, 6, 6, 6, 6, 8, 1, 8, 1, 6, 6, 6, 6, 6],
    [1, 8, 6, 6, 6, 6, 1, 8, 1, 1, 6, 6, 6, 6, 6],
    [1, 1, 3, 6, 6, 6, 8, 1, 8, 1, 6, 6, 6, 6, 6],
    [1, 1, 6, 3, 6, 3, 1, 1, 1, 8, 6, 6, 6, 6, 6],
    [1, 1, 6, 6, 3, 6, 1, 1, 1, 1, 3, 6, 6, 6, 3],
    [1, 1, 6, 3, 6, 3, 1, 1, 1, 1, 6, 3, 6, 3, 6],
    [1, 1, 3, 6, 6, 6, 8, 1, 1, 1, 6, 6, 3, 6, 6],
    [1, 8, 6, 6, 6, 6, 1, 8, 1, 1, 6, 3, 6, 3, 6],
    [8, 1, 6, 6, 6, 6, 1, 1, 8, 1, 3, 6, 6, 6, 3],
    [1, 1, 6, 6, 6, 6, 1, 1, 1, 8, 6, 6, 6, 6, 6],
    [1, 1, 6, 6, 6, 6, 1, 1, 8, 1, 3, 6, 6, 6, 6],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [8, 8, 8, 3, 3, 3, 3, 3, 3, 8, 4, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 3, 3, 3, 3, 3, 3, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 3, 3, 3, 3, 3, 3, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 3, 3, 3, 3, 3, 3, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [8, 8, 8, 3, 3, 3, 3, 3, 3, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 3, 3, 3, 3, 3, 3, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 3, 3, 3, 3, 3, 3, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 3, 3, 3, 3, 3, 3, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [8, 8, 8, 3, 3, 3, 3, 3, 3, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 3, 3, 3, 3, 3, 3, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
], dtype=int)

T_OUT = np.array([
    [8, 8, 4, 3, 3, 3, 3, 3, 3, 8, 4, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 1, 3, 3, 3, 3, 3, 4, 8, 4, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 3, 1, 3, 3, 3, 1, 8, 8, 8, 4, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 3, 3, 1, 3, 1, 3, 8, 8, 8, 8, 4, 8, 8, 8, 8, 8],
    [3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 1, 3, 1, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3],
    [3, 3, 3, 3, 1, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3],
    [8, 8, 8, 1, 3, 3, 3, 3, 3, 4, 8, 8, 8, 8, 8, 8, 8, 4, 8],
    [8, 8, 4, 3, 3, 3, 3, 3, 3, 8, 4, 8, 8, 8, 8, 8, 8, 8, 4],
    [8, 4, 8, 3, 3, 3, 3, 3, 3, 8, 8, 4, 8, 8, 8, 8, 8, 4, 8],
    [4, 8, 8, 3, 3, 3, 3, 3, 3, 8, 8, 8, 4, 8, 8, 8, 4, 8, 8],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 1, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 1, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 1, 3, 3],
    [8, 8, 8, 3, 3, 3, 3, 3, 3, 8, 8, 4, 8, 8, 8, 8, 8, 4, 8],
    [8, 8, 8, 3, 3, 3, 3, 3, 3, 8, 4, 8, 8, 8, 8, 8, 8, 8, 4],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(*args, **kwargs):
    raise NotImplementedError("Barnacles solution not available for 324")


# --- Code Golf Solution (Compressed) ---
def q(s):
    c = sum(s, [])
    t = c.count
    a = c[t(c[0]) < 4]
    o, f = ({}, {()})
    for d, e in enumerate(s):
        for m, i in enumerate(e):
            if t(i) < 4:
                o[a in e and a in c[m::len(e)]] = i
                f |= {d + m, (d - m,)}
    for d, e in enumerate(s):
        for m, i in enumerate(e):
            if {d + m, (d - m,)} & f:
                e[m] = o[a in e and a in c[m::len(e)]]
    return s


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, randint, sample, uniform

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

def generate_d07ae81c(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    lnf = lambda ij: shoot(ij, (1, 1)) | shoot(ij, (-1, -1)) | shoot(ij, (-1, 1)) | shoot(ij, (1, -1))
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    c1, c2, c3, c4 = sample(cols, 4)
    magiccol = 0
    gi = canvas(0, (h, w))
    ndivi = unifint(diff_lb, diff_ub, (1, (h * w) // 10))
    for k in range(ndivi):
        objs = objects(gi, T, F, F)
        objs = sfilter(objs, lambda o: min(shape(o)) > 3 and max(shape(o)) > 4)
        objs = sfilter(objs, lambda o: height(o) * width(o) == len(o))
        if len(objs) == 0:
            break
        obj = choice(totuple(objs))
        if choice((True, False)):
            loci = randint(uppermost(obj)+2, lowermost(obj)-1)
            newobj = backdrop(frozenset({(loci, leftmost(obj)), lrcorner(obj)}))
        else:
            locj = randint(leftmost(obj)+2, rightmost(obj)-1)
            newobj = backdrop(frozenset({(uppermost(obj), locj), lrcorner(obj)}))
        magiccol += 1
        gi = fill(gi, magiccol, newobj)
    objs = objects(gi, T, F, F)
    for ii, obj in enumerate(objs):
        col = c1 if ii == 0 else (c2 if ii == 1 else choice((c1, c2)))
        gi = fill(gi, col, toindices(obj))
    ofc1 = ofcolor(gi, c1)
    ofc2 = ofcolor(gi, c2)
    mn = min(len(ofc1), len(ofc2))
    n1 = unifint(diff_lb, diff_ub, (1, max(1, int(mn ** 0.5))))
    n2 = unifint(diff_lb, diff_ub, (1, max(1, int(mn ** 0.5))))
    srcs1 = set()
    for k in range(n1):
        cands = totuple((ofc1 - srcs1) - mapply(neighbors, srcs1))
        if len(cands) == 0:
            break
        srcs1.add(choice(cands))
    srcs2 = set()
    for k in range(n2):
        cands = totuple((ofc2 - srcs2) - mapply(neighbors, srcs2))
        if len(cands) == 0:
            break
        srcs2.add(choice(cands))
    gi = fill(gi, c3, srcs1)
    gi = fill(gi, c4, srcs2)
    lns = mapply(lnf, srcs1) | mapply(lnf, srcs2)
    ofc3 = ofc1 & lns
    ofc4 = ofc2 & lns
    go = fill(gi, c3, ofc3)
    go = fill(go, c4, ofc4)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
IntegerSet = FrozenSet[Integer]

ZERO = 0

UNITY = (1, 1)

NEG_UNITY = (-1, -1)

UP_RIGHT = (-1, 1)

DOWN_LEFT = (1, -1)

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

def size(
    container: Container
) -> Integer:
    """ cardinality """
    return len(container)

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

def palette(
    element: Element
) -> IntegerSet:
    """ colors occurring in object or grid """
    if isinstance(element, tuple):
        return frozenset({v for r in element for v in r})
    return frozenset({v for v, _ in element})

def toobject(
    patch: Patch,
    grid: Grid
) -> Object:
    """ object from patch and grid """
    h, w = len(grid), len(grid[0])
    return frozenset((grid[i][j], (i, j)) for i, j in toindices(patch) if 0 <= i < h and 0 <= j < w)

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_d07ae81c(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = lbind(ofcolor, I)
    x1 = lbind(mapply, neighbors)
    x2 = compose(x1, x0)
    x3 = fork(intersection, x0, x2)
    x4 = compose(size, x3)
    x5 = palette(I)
    x6 = matcher(x4, ZERO)
    x7 = sfilter(x5, x6)
    x8 = totuple(x7)
    x9 = first(x8)
    x10 = last(x8)
    x11 = ofcolor(I, x9)
    x12 = mapply(neighbors, x11)
    x13 = toobject(x12, I)
    x14 = mostcolor(x13)
    x15 = ofcolor(I, x10)
    x16 = mapply(neighbors, x15)
    x17 = toobject(x16, I)
    x18 = mostcolor(x17)
    x19 = rbind(shoot, UNITY)
    x20 = rbind(shoot, NEG_UNITY)
    x21 = fork(combine, x19, x20)
    x22 = rbind(shoot, UP_RIGHT)
    x23 = rbind(shoot, DOWN_LEFT)
    x24 = fork(combine, x22, x23)
    x25 = fork(combine, x21, x24)
    x26 = ofcolor(I, x10)
    x27 = ofcolor(I, x9)
    x28 = combine(x26, x27)
    x29 = mapply(x25, x28)
    x30 = ofcolor(I, x14)
    x31 = intersection(x30, x29)
    x32 = ofcolor(I, x18)
    x33 = intersection(x32, x29)
    x34 = fill(I, x9, x31)
    x35 = fill(x34, x10, x33)
    return x35


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_d07ae81c(inp)
        assert pred == _to_grid(expected), f"{name} failed"

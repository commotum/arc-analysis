# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "272f95fa"
SERIAL = "055"
URL    = "https://arcprize.org/play?task=272f95fa"

# --- Code Golf Concepts ---
CONCEPTS = [
    "detect_grid",
    "mimic_pattern",
    "grid_coloring",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [0, 0, 0, 0, 8, 2, 2, 2, 2, 2, 2, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 2, 2, 2, 2, 2, 2, 8, 0, 0, 0, 0, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [4, 4, 4, 4, 8, 6, 6, 6, 6, 6, 6, 8, 3, 3, 3, 3, 3, 3, 3],
    [4, 4, 4, 4, 8, 6, 6, 6, 6, 6, 6, 8, 3, 3, 3, 3, 3, 3, 3],
    [4, 4, 4, 4, 8, 6, 6, 6, 6, 6, 6, 8, 3, 3, 3, 3, 3, 3, 3],
    [4, 4, 4, 4, 8, 6, 6, 6, 6, 6, 6, 8, 3, 3, 3, 3, 3, 3, 3],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 0, 8, 1, 1, 1, 1, 1, 1, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 1, 1, 1, 1, 1, 1, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 1, 1, 1, 1, 1, 1, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 1, 1, 1, 1, 1, 1, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 1, 1, 1, 1, 1, 1, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 1, 1, 1, 1, 1, 1, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 1, 1, 1, 1, 1, 1, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 1, 1, 1, 1, 1, 1, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 1, 1, 1, 1, 1, 1, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 1, 1, 1, 1, 1, 1, 8, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E2_IN = np.array([
    [0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [0, 0, 8, 2, 2, 2, 2, 2, 2, 8, 0, 0, 0, 0],
    [0, 0, 8, 2, 2, 2, 2, 2, 2, 8, 0, 0, 0, 0],
    [0, 0, 8, 2, 2, 2, 2, 2, 2, 8, 0, 0, 0, 0],
    [0, 0, 8, 2, 2, 2, 2, 2, 2, 8, 0, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [4, 4, 8, 6, 6, 6, 6, 6, 6, 8, 3, 3, 3, 3],
    [4, 4, 8, 6, 6, 6, 6, 6, 6, 8, 3, 3, 3, 3],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 8, 1, 1, 1, 1, 1, 1, 8, 0, 0, 0, 0],
    [0, 0, 8, 1, 1, 1, 1, 1, 1, 8, 0, 0, 0, 0],
    [0, 0, 8, 1, 1, 1, 1, 1, 1, 8, 0, 0, 0, 0],
    [0, 0, 8, 1, 1, 1, 1, 1, 1, 8, 0, 0, 0, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0],
], dtype=int)

T_OUT = np.array([
    [0, 0, 0, 8, 2, 2, 2, 2, 8, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 2, 2, 2, 2, 8, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 2, 2, 2, 2, 8, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 2, 2, 2, 2, 8, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 2, 2, 2, 2, 8, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 2, 2, 2, 2, 8, 0, 0, 0, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [4, 4, 4, 8, 6, 6, 6, 6, 8, 3, 3, 3, 3, 3, 3],
    [4, 4, 4, 8, 6, 6, 6, 6, 8, 3, 3, 3, 3, 3, 3],
    [4, 4, 4, 8, 6, 6, 6, 6, 8, 3, 3, 3, 3, 3, 3],
    [4, 4, 4, 8, 6, 6, 6, 6, 8, 3, 3, 3, 3, 3, 3],
    [4, 4, 4, 8, 6, 6, 6, 6, 8, 3, 3, 3, 3, 3, 3],
    [4, 4, 4, 8, 6, 6, 6, 6, 8, 3, 3, 3, 3, 3, 3],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 8, 1, 1, 1, 1, 8, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 1, 1, 1, 1, 8, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 1, 1, 1, 1, 8, 0, 0, 0, 0, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j,A=range):
	c,E=len(j),len(j[0]);k=[A[:]for A in j];W,l=[A for A in A(c)if all(A==8 for A in j[A])];J,a=[C for C in A(E)if all(j[A][C]==8 for A in A(c))]
	for C in A(c):
		for e in A(E):
			if not k[C][e]:
				if C<W and J<e<a:k[C][e]=2
				elif W<C<l and e<J:k[C][e]=4
				elif W<C<l and J<e<a:k[C][e]=6
				elif W<C<l and e>a:k[C][e]=3
				elif C>l and J<e<a:k[C][e]=1
	return k


# --- Code Golf Solution (Compressed) ---
def q(i, z=0):
    return i * 0 != 0 and [p(y, 3 * (z := (z + ([y] > i)))) for y in i] or i or 2222096 >> z & 7


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, sample, shuffle, uniform

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

F = False

T = True

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

def argmax(
    container: Container,
    compfunc: Callable
) -> Any:
    """ largest item by custom order """
    return max(container, key=compfunc, default=None)

def argmin(
    container: Container,
    compfunc: Callable
) -> Any:
    """ smallest item by custom order """
    return min(container, key=compfunc, default=None)

def sfilter(
    container: Container,
    condition: Callable
) -> Container:
    """ keep elements in container that satisfy condition """
    return type(container)(e for e in container if condition(e))

def extract(
    container: Container,
    condition: Callable
) -> Any:
    """ first element of container that satisfies condition """
    return next(e for e in container if condition(e))

def interval(
    start: Integer,
    stop: Integer,
    step: Integer
) -> Tuple:
    """ range """
    return tuple(range(start, stop, step))

def mostcolor(
    element: Element
) -> Integer:
    """ most common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return max(set(values), key=values.count)

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

def bordering(
    patch: Patch,
    grid: Grid
) -> Boolean:
    """ whether a patch is adjacent to a grid border """
    return uppermost(patch) == 0 or leftmost(patch) == 0 or lowermost(patch) == len(grid) - 1 or rightmost(patch) == len(grid[0]) - 1

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

def corners(
    patch: Patch
) -> Indices:
    """ indices of corners """
    return frozenset({ulcorner(patch), urcorner(patch), llcorner(patch), lrcorner(patch)})

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

def generate_272f95fa(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2, 3, 4, 6))    
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    bgc, linc = sample(cols, 2)
    c = canvas(bgc, (5, 5))
    l1 = connect((1, 0), (1, 4))
    l2 = connect((3, 0), (3, 4))
    lns = l1 | l2
    gi = fill(dmirror(fill(c, linc, lns)), linc, lns)
    hdist = [0, 0, 0]
    wdist = [0, 0, 0]
    idx = 0
    for k in range(h - 2):
        hdist[idx] += 1
        idx = (idx + 1) % 3
    for k in range(w - 2):
        wdist[idx] += 1
        idx = (idx + 1) % 3
    shuffle(hdist)
    shuffle(wdist)
    hdelt1 = unifint(diff_lb, diff_ub, (0, hdist[0] - 1))
    hdist[0] -= hdelt1
    hdist[1] += hdelt1
    hdelt2 = unifint(diff_lb, diff_ub, (0, min(hdist[1], hdist[2]) - 1))
    hdelt2 = choice((+hdelt2, -hdelt2))
    hdist[1] += hdelt2
    hdist[2] -= hdelt2
    wdelt1 = unifint(diff_lb, diff_ub, (0, wdist[0] - 1))
    wdist[0] -= wdelt1
    wdist[1] += wdelt1
    wdelt2 = unifint(diff_lb, diff_ub, (0, min(wdist[1], wdist[2]) - 1))
    wdelt2 = choice((+wdelt2, -wdelt2))
    wdist[1] += wdelt2
    wdist[2] -= wdelt2
    gi = gi[:1] * hdist[0] + gi[1:2] + gi[2:3] * hdist[1] + gi[3:4] + gi[4:5] * hdist[2]
    gi = dmirror(gi)
    gi = gi[:1] * wdist[0] + gi[1:2] + gi[2:3] * wdist[1] + gi[3:4] + gi[4:5] * wdist[2]
    gi = dmirror(gi)
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
    objs = objects(gi, T, T, F)
    bgobjs = colorfilter(objs, bgc)
    cnrs = corners(asindices(gi))
    bgobjs = sfilter(bgobjs, lambda o: len(toindices(o) & cnrs) == 0)
    pinkobj = extract(bgobjs, lambda o: not bordering(o, gi))
    yellobj = argmin(bgobjs, leftmost)
    greenobj = argmax(bgobjs, rightmost)
    redobj = argmin(bgobjs, uppermost)
    blueobj = argmax(bgobjs, lowermost)
    go = fill(gi, 6, pinkobj)
    go = fill(go, 4, yellobj)
    go = fill(go, 3, greenobj)
    go = fill(go, 2, redobj)
    go = fill(go, 1, blueobj)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
ONE = 1

TWO = 2

THREE = 3

FOUR = 4

SIX = 6

ORIGIN = (0, 0)

def flip(
    b: Boolean
) -> Boolean:
    """ logical not """
    return not b

def remove(
    value: Any,
    container: Container
) -> Container:
    """ remove item from container """
    return type(container)(e for e in container if e != value)

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

def hmatching(
    a: Patch,
    b: Patch
) -> Boolean:
    """ whether there exists a row for which both patches have cells """
    return len(set(i for i, j in toindices(a)) & set(i for i, j in toindices(b))) > 0

def vmatching(
    a: Patch,
    b: Patch
) -> Boolean:
    """ whether there exists a column for which both patches have cells """
    return len(set(j for i, j in toindices(a)) & set(j for i, j in toindices(b))) > 0

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

def verify_272f95fa(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = objects(I, T, F, F)
    x1 = index(I, ORIGIN)
    x2 = colorfilter(x0, x1)
    x3 = apply(toindices, x2)
    x4 = rbind(bordering, I)
    x5 = compose(flip, x4)
    x6 = extract(x3, x5)
    x7 = remove(x6, x3)
    x8 = lbind(vmatching, x6)
    x9 = lbind(hmatching, x6)
    x10 = sfilter(x7, x8)
    x11 = sfilter(x7, x9)
    x12 = argmin(x10, uppermost)
    x13 = argmax(x10, uppermost)
    x14 = argmin(x11, leftmost)
    x15 = argmax(x11, leftmost)
    x16 = fill(I, SIX, x6)
    x17 = fill(x16, TWO, x12)
    x18 = fill(x17, ONE, x13)
    x19 = fill(x18, FOUR, x14)
    x20 = fill(x19, THREE, x15)
    return x20


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_272f95fa(inp)
        assert pred == _to_grid(expected), f"{name} failed"

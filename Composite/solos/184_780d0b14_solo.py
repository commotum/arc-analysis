# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "780d0b14"
SERIAL = "184"
URL    = "https://arcprize.org/play?task=780d0b14"

# --- Code Golf Concepts ---
CONCEPTS = [
    "detect_grid",
    "summarize",
]

# --- Example Grids ---
E1_IN = np.array([
    [1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 8, 8, 8, 8, 8, 0, 8, 8, 8],
    [1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 8, 0, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 1, 1, 0, 1, 1, 1, 1, 0, 8, 0, 8, 8, 0, 8, 8, 8, 0, 8, 8],
    [1, 0, 1, 1, 1, 1, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 8],
    [1, 1, 0, 1, 1, 1, 1, 1, 0, 8, 8, 8, 0, 8, 8, 8, 0, 8, 0, 0],
    [1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 8, 8, 0, 8, 8, 8, 0, 0, 0, 8],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [6, 6, 6, 6, 6, 0, 6, 6, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0],
    [6, 6, 6, 6, 6, 6, 6, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0],
    [0, 6, 0, 6, 6, 6, 0, 6, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
    [6, 6, 6, 0, 6, 6, 6, 6, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1],
    [6, 0, 6, 6, 0, 6, 0, 6, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1],
    [6, 6, 6, 6, 6, 0, 6, 6, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
    [6, 6, 6, 6, 6, 0, 6, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
    [6, 6, 6, 0, 6, 6, 0, 6, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
    [0, 6, 6, 6, 0, 0, 6, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0],
    [6, 0, 0, 0, 6, 0, 6, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
    [6, 6, 0, 6, 0, 6, 6, 6, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0],
], dtype=int)

E1_OUT = np.array([
    [1, 8],
    [6, 1],
], dtype=int)

E2_IN = np.array([
    [4, 4, 4, 4, 4, 0, 0, 8, 0, 8, 8, 8, 0, 0, 3, 3, 3, 0, 0, 3, 3, 3],
    [4, 4, 4, 0, 0, 4, 0, 8, 8, 8, 8, 8, 0, 0, 3, 3, 3, 3, 0, 3, 3, 0],
    [4, 4, 4, 4, 0, 0, 0, 8, 8, 0, 0, 8, 0, 0, 3, 3, 3, 0, 3, 0, 3, 3],
    [4, 4, 0, 0, 4, 4, 0, 8, 8, 8, 8, 8, 8, 0, 3, 3, 3, 3, 0, 3, 3, 3],
    [4, 4, 4, 4, 4, 4, 0, 0, 8, 8, 8, 8, 8, 0, 3, 0, 3, 0, 3, 0, 3, 0],
    [0, 0, 4, 4, 4, 4, 0, 8, 0, 8, 0, 8, 0, 0, 3, 0, 3, 3, 3, 3, 3, 3],
    [4, 4, 0, 4, 4, 0, 0, 8, 8, 8, 8, 0, 8, 0, 3, 0, 0, 3, 3, 3, 3, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 2, 0, 2, 2, 2, 2, 0, 8, 0, 8, 0, 0, 8, 8, 8],
    [1, 0, 1, 1, 0, 1, 0, 2, 0, 2, 2, 2, 0, 0, 8, 8, 8, 0, 0, 8, 8, 8],
    [1, 1, 1, 0, 1, 0, 0, 2, 0, 2, 2, 2, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8],
    [1, 1, 0, 1, 0, 1, 0, 2, 2, 2, 2, 0, 2, 0, 0, 0, 8, 8, 8, 0, 8, 8],
    [1, 1, 1, 0, 1, 0, 0, 2, 2, 0, 2, 2, 0, 0, 0, 8, 0, 8, 8, 8, 8, 0],
    [1, 1, 1, 1, 1, 1, 0, 0, 2, 2, 2, 0, 2, 0, 8, 8, 0, 0, 8, 0, 8, 8],
    [1, 1, 1, 0, 0, 0, 0, 2, 0, 2, 2, 2, 2, 0, 8, 8, 0, 0, 0, 8, 8, 8],
    [1, 0, 0, 1, 0, 1, 0, 2, 2, 0, 2, 2, 0, 0, 8, 0, 8, 8, 0, 0, 0, 8],
    [1, 1, 1, 1, 0, 1, 0, 0, 2, 2, 2, 0, 2, 0, 0, 8, 8, 0, 0, 0, 8, 0],
    [1, 1, 0, 1, 1, 1, 0, 2, 2, 2, 0, 2, 0, 0, 8, 0, 8, 8, 0, 0, 8, 8],
], dtype=int)

E2_OUT = np.array([
    [4, 8, 3],
    [1, 2, 8],
], dtype=int)

E3_IN = np.array([
    [2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 7, 0, 0, 7, 0, 0],
    [2, 2, 0, 0, 2, 0, 2, 0, 7, 0, 7, 0, 7, 7, 7, 7, 0],
    [2, 2, 2, 2, 0, 2, 2, 0, 0, 7, 7, 0, 0, 7, 7, 0, 7],
    [2, 0, 2, 2, 0, 2, 2, 0, 0, 0, 7, 7, 7, 7, 7, 7, 0],
    [2, 2, 2, 0, 2, 2, 2, 0, 0, 7, 0, 7, 7, 7, 0, 0, 0],
    [2, 0, 2, 0, 2, 2, 2, 0, 7, 7, 0, 7, 7, 0, 0, 7, 7],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 4, 4, 4, 4, 4, 0, 0, 0, 8, 0, 8, 8, 8, 8, 8, 8],
    [4, 0, 4, 4, 0, 4, 0, 0, 8, 0, 8, 8, 8, 8, 8, 8, 8],
    [4, 0, 0, 4, 0, 4, 4, 0, 0, 8, 0, 8, 8, 0, 8, 0, 8],
    [4, 4, 0, 0, 0, 0, 4, 0, 8, 8, 0, 8, 8, 8, 8, 8, 8],
    [4, 4, 4, 4, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 0],
    [4, 4, 4, 4, 0, 4, 4, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [4, 4, 4, 4, 4, 4, 0, 0, 8, 8, 8, 0, 0, 8, 8, 8, 0],
    [0, 4, 4, 4, 0, 4, 4, 0, 8, 8, 0, 8, 8, 8, 8, 0, 8],
    [0, 0, 0, 0, 4, 4, 4, 0, 0, 8, 0, 0, 8, 0, 8, 8, 8],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0, 6, 6, 0, 6, 6, 0, 6, 6, 6],
    [0, 1, 1, 1, 1, 1, 0, 0, 6, 6, 6, 0, 6, 6, 6, 6, 0],
    [1, 1, 1, 1, 1, 0, 1, 0, 6, 6, 6, 6, 0, 6, 6, 6, 6],
    [1, 0, 0, 0, 1, 1, 1, 0, 6, 6, 6, 0, 6, 6, 6, 6, 6],
    [1, 0, 1, 1, 1, 0, 0, 0, 6, 6, 6, 6, 6, 0, 0, 6, 6],
    [1, 1, 1, 1, 1, 1, 1, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6],
], dtype=int)

E3_OUT = np.array([
    [2, 7],
    [4, 8],
    [1, 6],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [3, 3, 3, 0, 3, 3, 3, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4],
    [3, 3, 3, 3, 3, 3, 3, 0, 2, 2, 0, 2, 2, 2, 2, 0, 4, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [0, 3, 0, 0, 3, 3, 0, 0, 2, 2, 0, 0, 2, 2, 2, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [3, 0, 3, 3, 3, 3, 3, 0, 2, 0, 2, 2, 2, 2, 2, 0, 4, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 8, 8, 8, 0, 8, 8, 8, 8, 8, 0, 8, 0],
    [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 8, 8, 8, 8, 8, 8, 0, 8, 0, 8, 8, 0],
    [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 8, 8, 8, 8, 8, 8, 8, 0, 8, 8, 8, 8],
    [1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 8, 8, 8, 0, 8, 8, 8, 8, 8, 8, 0, 8],
    [0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 8, 8, 8, 0, 8, 8, 0, 8, 8, 8],
    [1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 8, 8, 8, 8, 8, 8, 0, 8, 0, 8, 8, 0],
    [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 8, 8, 8, 8, 8, 0, 0, 0, 8, 8, 8, 8],
    [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 8, 8, 8, 8, 0, 8, 8, 8, 8, 8, 0, 8],
    [1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 8, 0],
    [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 8, 0, 8, 0, 8, 8, 8, 8, 8, 8, 8, 8],
    [1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 8, 8, 8, 8, 8, 8, 0, 8, 0, 8, 8, 8],
    [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 8, 0, 8, 8, 8, 8, 8, 8, 8, 8, 0, 8],
    [0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 8, 0, 8, 8, 8, 8, 8, 0, 8, 8, 8, 8],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [7, 7, 0, 7, 7, 0, 7, 0, 3, 3, 0, 0, 3, 3, 3, 0, 2, 0, 2, 2, 2, 2, 0, 2, 2, 0, 2, 2],
    [7, 7, 7, 0, 7, 7, 7, 0, 0, 3, 3, 0, 3, 0, 0, 0, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 0],
    [7, 7, 7, 7, 7, 7, 7, 0, 3, 3, 3, 3, 3, 3, 3, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
], dtype=int)

T_OUT = np.array([
    [3, 2, 4],
    [1, 1, 8],
    [7, 3, 2],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
	A=range;c,E=len(j),len(j[0]);k=[k for k in A(c)if all(j[k][A]==0 for A in A(E))];W=[k for k in A(E)if all(j[A][k]==0 for A in A(c))];k=[-1]+k+[c];W=[-1]+W+[E];l=[]
	for J in A(len(k)-1):
		a=[]
		for C in A(len(W)-1):
			for e in A(k[J]+1,k[J+1]):
				for K in A(W[C]+1,W[C+1]):
					if j[e][K]:a.append(j[e][K]);break
				else:continue
				break
		if a:l.append(a)
	return l


# --- Code Golf Solution (Compressed) ---
def q(g, *h, q=[]):
    return [q + 0 * (q := r) for *r, in zip(*(h or p(*g))) if [0, *r, (q := [*map(max, q + r, r)])] > r] + [q]


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

def dedupe(
    iterable: Tuple
) -> Tuple:
    """ remove duplicates """
    return tuple(e for i, e in enumerate(iterable) if iterable.index(e) == i)

def repeat(
    item: Any,
    num: Integer
) -> Tuple:
    """ repetition of item within vector """
    return tuple(item for i in range(num))

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

def generate_780d0b14(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    nh = unifint(diff_lb, diff_ub, (2, 6))
    nw = unifint(diff_lb, diff_ub, (2, 6))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    ncols = unifint(diff_lb, diff_ub, (3, 9))
    ccols = sample(remcols, ncols)
    go = canvas(-1, (nh, nw))
    obj = {(choice(ccols), ij) for ij in asindices(go)}
    go = paint(go, obj)
    while len(dedupe(go)) < nh or len(dedupe(dmirror(go))) < nw:
        obj = {(choice(ccols), ij) for ij in asindices(go)}
        go = paint(go, obj)
    h = unifint(diff_lb, diff_ub, (2*nh+nh-1, 30))
    w = unifint(diff_lb, diff_ub, (2*nw+nw-1, 30))
    hdist = [2 for k in range(nh)]
    for k in range(h - 2 * nh - nh + 1):
        idx = randint(0, nh - 1)
        hdist[idx] += 1
    wdist = [2 for k in range(nw)]
    for k in range(w - 2 * nw - nw + 1):
        idx = randint(0, nw - 1)
        wdist[idx] += 1
    gi = merge(tuple(repeat(r, c) + (repeat(bgc, nw),) for r, c in zip(go, hdist)))[:-1]
    gi = dmirror(merge(tuple(repeat(r, c) + (repeat(bgc, h),) for r, c in zip(dmirror(gi), wdist)))[:-1])
    objs = objects(gi, T, F, F)
    bgobjs = colorfilter(objs, bgc)
    objs = objs - bgobjs
    for obj in objs:
        gi = fill(gi, bgc, sample(totuple(toindices(obj)), unifint(diff_lb, diff_ub, (1, len(obj) // 2))))
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
IntegerSet = FrozenSet[Integer]

NEG_ONE = -1

def first(
    container: Container
) -> Any:
    """ first item of container """
    return next(iter(container))

def other(
    container: Container,
    value: Any
) -> Any:
    """ other value in the container """
    return first(remove(value, container))

def astuple(
    a: Integer,
    b: Integer
) -> IntegerTuple:
    """ constructs a tuple """
    return (a, b)

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

def color(
    obj: Object
) -> Integer:
    """ color of object """
    return next(iter(obj))[0]

def hconcat(
    a: Grid,
    b: Grid
) -> Grid:
    """ concatenate two grids horizontally """
    return tuple(i + j for i, j in zip(a, b))

def frontiers(
    grid: Grid
) -> Objects:
    """ set of frontiers """
    h, w = len(grid), len(grid[0])
    row_indices = tuple(i for i, r in enumerate(grid) if len(set(r)) == 1)
    column_indices = tuple(j for j, c in enumerate(dmirror(grid)) if len(set(c)) == 1)
    hfrontiers = frozenset({frozenset({(grid[i][j], (i, j)) for j in range(w)}) for i in row_indices})
    vfrontiers = frozenset({frozenset({(grid[i][j], (i, j)) for i in range(h)}) for j in column_indices})
    return hfrontiers | vfrontiers

def compress(
    grid: Grid
) -> Grid:
    """ removes frontiers from grid """
    ri = tuple(i for i, r in enumerate(grid) if len(set(r)) == 1)
    ci = tuple(j for j, c in enumerate(dmirror(grid)) if len(set(c)) == 1)
    return tuple(tuple(v for j, v in enumerate(r) if j not in ci) for i, r in enumerate(grid) if i not in ri)

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_780d0b14(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = frontiers(I)
    x1 = merge(x0)
    x2 = color(x1)
    x3 = merge(x0)
    x4 = fill(I, NEG_ONE, x3)
    x5 = shape(I)
    x6 = canvas(NEG_ONE, x5)
    x7 = hconcat(x4, x6)
    x8 = objects(x7, F, F, T)
    x9 = rbind(other, x2)
    x10 = compose(x9, palette)
    x11 = fork(astuple, x10, ulcorner)
    x12 = apply(x11, x8)
    x13 = merge(x8)
    x14 = fill(I, x2, x13)
    x15 = paint(x14, x12)
    x16 = compress(x15)
    return x16


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_780d0b14(inp)
        assert pred == _to_grid(expected), f"{name} failed"

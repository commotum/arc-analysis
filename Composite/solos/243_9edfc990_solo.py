# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "9edfc990"
SERIAL = "243"
URL    = "https://arcprize.org/play?task=9edfc990"

# --- Code Golf Concepts ---
CONCEPTS = [
    "background_filling",
    "holes",
]

# --- Example Grids ---
E1_IN = np.array([
    [9, 0, 0, 0, 0, 2, 8, 0, 9, 0, 2, 0, 9],
    [1, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 9, 5],
    [9, 0, 4, 9, 3, 0, 0, 5, 7, 0, 8, 0, 8],
    [0, 0, 8, 6, 0, 6, 0, 1, 0, 0, 0, 4, 1],
    [3, 6, 0, 1, 0, 3, 9, 0, 0, 4, 5, 7, 2],
    [0, 8, 0, 0, 0, 0, 0, 0, 7, 1, 8, 0, 0],
    [9, 0, 0, 2, 0, 0, 0, 7, 5, 7, 0, 8, 4],
    [0, 0, 0, 8, 7, 5, 0, 0, 7, 0, 0, 5, 0],
    [9, 9, 0, 0, 0, 0, 5, 0, 0, 5, 0, 0, 0],
    [8, 0, 0, 8, 0, 6, 5, 0, 0, 0, 0, 9, 0],
    [4, 0, 0, 6, 0, 7, 9, 9, 8, 0, 5, 7, 3],
    [0, 0, 0, 0, 0, 0, 0, 7, 2, 0, 0, 0, 8],
    [0, 0, 0, 7, 5, 0, 5, 0, 0, 0, 0, 0, 3],
], dtype=int)

E1_OUT = np.array([
    [9, 1, 1, 1, 1, 2, 8, 1, 9, 1, 2, 0, 9],
    [1, 1, 1, 6, 1, 1, 1, 1, 1, 1, 1, 9, 5],
    [9, 1, 4, 9, 3, 1, 1, 5, 7, 1, 8, 0, 8],
    [1, 1, 8, 6, 1, 6, 1, 1, 1, 1, 1, 4, 1],
    [3, 6, 1, 1, 1, 3, 9, 1, 1, 4, 5, 7, 2],
    [0, 8, 1, 1, 1, 1, 1, 1, 7, 1, 8, 0, 0],
    [9, 1, 1, 2, 1, 1, 1, 7, 5, 7, 1, 8, 4],
    [1, 1, 1, 8, 7, 5, 1, 1, 7, 1, 1, 5, 1],
    [9, 9, 1, 1, 1, 1, 5, 1, 1, 5, 1, 1, 1],
    [8, 1, 1, 8, 1, 6, 5, 1, 1, 1, 1, 9, 1],
    [4, 1, 1, 6, 1, 7, 9, 9, 8, 1, 5, 7, 3],
    [1, 1, 1, 1, 1, 1, 1, 7, 2, 1, 1, 1, 8],
    [1, 1, 1, 7, 5, 1, 5, 1, 1, 1, 1, 1, 3],
], dtype=int)

E2_IN = np.array([
    [0, 0, 2, 0, 9, 6, 5, 5, 5, 0, 2, 1, 0, 0, 0],
    [3, 0, 4, 4, 9, 0, 0, 0, 3, 9, 0, 0, 0, 5, 0],
    [8, 9, 2, 0, 1, 0, 6, 8, 0, 0, 0, 8, 0, 8, 0],
    [6, 0, 4, 0, 4, 0, 0, 1, 6, 1, 6, 9, 1, 4, 2],
    [7, 7, 7, 3, 0, 0, 6, 4, 0, 4, 0, 1, 3, 0, 0],
    [7, 6, 0, 4, 0, 2, 0, 0, 4, 0, 8, 0, 0, 7, 6],
    [0, 0, 4, 7, 8, 3, 0, 4, 0, 0, 5, 0, 6, 0, 3],
    [0, 8, 0, 0, 2, 0, 0, 0, 1, 0, 2, 0, 0, 1, 0],
    [3, 3, 1, 0, 2, 0, 0, 6, 0, 8, 6, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 8, 0, 0, 0, 9, 0, 4, 0, 2, 8],
    [0, 0, 4, 1, 4, 9, 0, 7, 0, 1, 0, 5, 0, 0, 8],
    [7, 2, 0, 0, 4, 5, 1, 0, 9, 0, 0, 6, 4, 0, 0],
    [0, 0, 0, 0, 9, 6, 3, 1, 3, 3, 9, 0, 0, 0, 5],
    [0, 5, 0, 4, 0, 7, 9, 9, 0, 0, 0, 0, 9, 4, 0],
    [0, 9, 8, 8, 0, 6, 8, 0, 0, 0, 8, 0, 0, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [0, 0, 2, 0, 9, 6, 5, 5, 5, 0, 2, 1, 1, 1, 1],
    [3, 0, 4, 4, 9, 1, 1, 1, 3, 9, 1, 1, 1, 5, 1],
    [8, 9, 2, 1, 1, 1, 6, 8, 1, 1, 1, 8, 1, 8, 1],
    [6, 0, 4, 1, 4, 1, 1, 1, 6, 1, 6, 9, 1, 4, 2],
    [7, 7, 7, 3, 1, 1, 6, 4, 0, 4, 1, 1, 3, 0, 0],
    [7, 6, 0, 4, 1, 2, 1, 1, 4, 1, 8, 1, 1, 7, 6],
    [0, 0, 4, 7, 8, 3, 1, 4, 1, 1, 5, 1, 6, 1, 3],
    [0, 8, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1],
    [3, 3, 1, 1, 2, 1, 1, 6, 1, 8, 6, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 8, 1, 1, 1, 9, 1, 4, 1, 2, 8],
    [1, 1, 4, 1, 4, 9, 1, 7, 1, 1, 1, 5, 1, 1, 8],
    [7, 2, 1, 1, 4, 5, 1, 1, 9, 1, 1, 6, 4, 1, 1],
    [1, 1, 1, 1, 9, 6, 3, 1, 3, 3, 9, 1, 1, 1, 5],
    [1, 5, 1, 4, 0, 7, 9, 9, 1, 1, 1, 1, 9, 4, 1],
    [1, 9, 8, 8, 0, 6, 8, 1, 1, 1, 8, 1, 1, 1, 1],
], dtype=int)

E3_IN = np.array([
    [7, 4, 4, 0, 4, 0, 0, 6, 1, 1, 1, 0, 0, 6, 0, 5],
    [1, 1, 3, 3, 4, 0, 3, 8, 5, 3, 4, 5, 0, 8, 2, 8],
    [8, 0, 4, 8, 8, 5, 0, 9, 0, 0, 0, 5, 5, 8, 5, 8],
    [0, 2, 6, 0, 0, 0, 0, 3, 0, 1, 0, 8, 0, 4, 0, 8],
    [8, 0, 2, 8, 0, 7, 0, 0, 0, 9, 0, 7, 3, 0, 3, 6],
    [0, 0, 0, 0, 0, 0, 5, 3, 0, 6, 0, 6, 0, 4, 5, 7],
    [6, 6, 0, 0, 3, 1, 0, 0, 2, 5, 0, 0, 0, 3, 4, 5],
    [7, 0, 7, 8, 0, 1, 0, 0, 0, 9, 0, 7, 3, 0, 3, 0],
    [0, 6, 0, 0, 5, 6, 6, 5, 9, 8, 3, 9, 0, 7, 0, 0],
    [7, 5, 0, 0, 0, 8, 0, 6, 9, 0, 0, 7, 1, 0, 0, 0],
    [6, 5, 3, 4, 3, 0, 6, 9, 4, 1, 8, 9, 2, 8, 7, 7],
    [8, 6, 8, 6, 3, 2, 7, 3, 0, 2, 0, 0, 2, 1, 0, 0],
    [9, 0, 0, 0, 6, 1, 8, 0, 3, 3, 0, 2, 0, 2, 1, 4],
    [0, 4, 0, 0, 0, 0, 1, 0, 0, 0, 6, 0, 4, 4, 5, 6],
    [0, 5, 0, 8, 3, 2, 1, 0, 5, 9, 1, 8, 7, 0, 2, 7],
    [0, 9, 0, 1, 8, 6, 0, 9, 9, 8, 0, 9, 0, 0, 3, 0],
], dtype=int)

E3_OUT = np.array([
    [7, 4, 4, 0, 4, 0, 0, 6, 1, 1, 1, 1, 1, 6, 0, 5],
    [1, 1, 3, 3, 4, 0, 3, 8, 5, 3, 4, 5, 1, 8, 2, 8],
    [8, 1, 4, 8, 8, 5, 1, 9, 1, 1, 1, 5, 5, 8, 5, 8],
    [0, 2, 6, 1, 1, 1, 1, 3, 1, 1, 1, 8, 0, 4, 0, 8],
    [8, 1, 2, 8, 1, 7, 1, 1, 1, 9, 1, 7, 3, 0, 3, 6],
    [1, 1, 1, 1, 1, 1, 5, 3, 1, 6, 1, 6, 1, 4, 5, 7],
    [6, 6, 1, 1, 3, 1, 1, 1, 2, 5, 1, 1, 1, 3, 4, 5],
    [7, 0, 7, 8, 1, 1, 1, 1, 1, 9, 1, 7, 3, 0, 3, 1],
    [0, 6, 0, 0, 5, 6, 6, 5, 9, 8, 3, 9, 1, 7, 1, 1],
    [7, 5, 0, 0, 0, 8, 0, 6, 9, 1, 1, 7, 1, 1, 1, 1],
    [6, 5, 3, 4, 3, 0, 6, 9, 4, 1, 8, 9, 2, 8, 7, 7],
    [8, 6, 8, 6, 3, 2, 7, 3, 0, 2, 0, 0, 2, 1, 1, 1],
    [9, 1, 1, 1, 6, 1, 8, 1, 3, 3, 0, 2, 0, 2, 1, 4],
    [0, 4, 1, 1, 1, 1, 1, 1, 1, 1, 6, 0, 4, 4, 5, 6],
    [0, 5, 1, 8, 3, 2, 1, 1, 5, 9, 1, 8, 7, 0, 2, 7],
    [0, 9, 1, 1, 8, 6, 1, 9, 9, 8, 1, 9, 0, 0, 3, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 0, 8, 0, 5, 0, 0, 9, 0, 6, 0, 0, 0, 0, 5],
    [6, 7, 6, 0, 4, 0, 2, 0, 0, 8, 3, 6, 2, 0, 0, 0],
    [0, 0, 0, 7, 0, 0, 5, 4, 1, 0, 1, 7, 6, 0, 0, 0],
    [0, 5, 8, 0, 9, 0, 0, 2, 2, 0, 8, 0, 4, 0, 0, 7],
    [4, 0, 0, 4, 2, 2, 7, 3, 2, 0, 6, 4, 9, 9, 9, 0],
    [0, 1, 8, 0, 5, 0, 0, 0, 2, 0, 0, 8, 0, 9, 6, 6],
    [9, 9, 0, 2, 8, 0, 0, 3, 0, 0, 2, 0, 0, 5, 8, 0],
    [1, 3, 0, 1, 6, 1, 0, 0, 0, 8, 0, 0, 0, 4, 0, 0],
    [0, 0, 4, 0, 7, 4, 0, 0, 4, 0, 0, 5, 8, 0, 4, 0],
    [0, 0, 0, 6, 0, 6, 0, 0, 0, 0, 0, 8, 0, 1, 4, 4],
    [0, 9, 0, 0, 9, 0, 0, 0, 0, 0, 1, 5, 0, 6, 0, 0],
    [6, 0, 7, 5, 9, 0, 7, 0, 0, 0, 4, 6, 0, 2, 8, 0],
    [5, 0, 0, 0, 1, 0, 2, 4, 8, 0, 0, 3, 0, 9, 0, 8],
    [1, 0, 0, 2, 4, 0, 0, 0, 1, 7, 0, 0, 0, 0, 5, 0],
    [6, 9, 0, 0, 7, 7, 1, 0, 2, 0, 0, 9, 1, 0, 3, 0],
    [1, 8, 3, 0, 0, 9, 7, 0, 2, 7, 2, 0, 8, 9, 0, 0],
], dtype=int)

T_OUT = np.array([
    [0, 0, 0, 8, 0, 5, 1, 1, 9, 0, 6, 0, 0, 0, 0, 5],
    [6, 7, 6, 0, 4, 0, 2, 1, 1, 8, 3, 6, 2, 0, 0, 0],
    [0, 0, 0, 7, 0, 0, 5, 4, 1, 1, 1, 7, 6, 0, 0, 0],
    [0, 5, 8, 0, 9, 0, 0, 2, 2, 1, 8, 0, 4, 0, 0, 7],
    [4, 1, 1, 4, 2, 2, 7, 3, 2, 1, 6, 4, 9, 9, 9, 0],
    [1, 1, 8, 0, 5, 1, 1, 1, 2, 1, 1, 8, 1, 9, 6, 6],
    [9, 9, 1, 2, 8, 1, 1, 3, 1, 1, 2, 1, 1, 5, 8, 0],
    [1, 3, 1, 1, 6, 1, 1, 1, 1, 8, 1, 1, 1, 4, 0, 0],
    [1, 1, 4, 1, 7, 4, 1, 1, 4, 1, 1, 5, 8, 1, 4, 0],
    [1, 1, 1, 6, 0, 6, 1, 1, 1, 1, 1, 8, 1, 1, 4, 4],
    [1, 9, 1, 1, 9, 1, 1, 1, 1, 1, 1, 5, 1, 6, 0, 0],
    [6, 1, 7, 5, 9, 1, 7, 1, 1, 1, 4, 6, 1, 2, 8, 0],
    [5, 1, 1, 1, 1, 1, 2, 4, 8, 1, 1, 3, 1, 9, 0, 8],
    [1, 1, 1, 2, 4, 1, 1, 1, 1, 7, 1, 1, 1, 1, 5, 0],
    [6, 9, 1, 1, 7, 7, 1, 1, 2, 1, 1, 9, 1, 1, 3, 0],
    [1, 8, 3, 1, 1, 9, 7, 1, 2, 7, 2, 0, 8, 9, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(g,L=len,R=range):
 h,w=L(g),L(g[0])
 for z in R(25):
  for r in R(h):
   for c in R(w):
    if g[r][c]==0:
     if c+1<w:
      if g[r][c+1]==1:g[r][c]=1
     if r+1<h:
      if g[r+1][c]==1:g[r][c]=1
     if c-1>=0:
      if g[r][c-1]==1:g[r][c]=1
     if r-1>=0:
      if g[r-1][c]==1:g[r][c]=1
 return g


# --- Code Golf Solution (Compressed) ---
def q(g):
    return exec('g[::-1]=zip(*eval(str(g).replace("1, 0","1,1")));' * 80) or g


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, sample, uniform

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

def interval(
    start: Integer,
    stop: Integer,
    step: Integer
) -> Tuple:
    """ range """
    return tuple(range(start, stop, step))

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

def generate_9edfc990(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(2, 10, 1)
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    namt = unifint(diff_lb, diff_ub, (int(0.4 * h * w), int(0.7 * h * w)))
    gi = canvas(0, (h, w))
    inds = asindices(gi)
    locs = sample(totuple(inds), namt)
    noise = {(choice(cols), ij) for ij in locs}
    gi = paint(gi, noise)
    remlocs = inds - set(locs)
    numc = unifint(diff_lb, diff_ub, (1, max(1, len(remlocs) // 10)))
    blocs = sample(totuple(remlocs), numc)
    gi = fill(gi, 1, blocs)
    objs = objects(gi, T, F, F)
    objs = colorfilter(objs, 0)
    res = mfilter(objs, rbind(adjacent, blocs))
    go = fill(gi, 1, res)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
ZERO = 0

ONE = 1

def ofcolor(
    grid: Grid,
    value: Integer
) -> Indices:
    """ indices of all grid cells with value """
    return frozenset((i, j) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value)

def recolor(
    value: Integer,
    patch: Patch
) -> Object:
    """ recolor patch """
    return frozenset((value, index) for index in toindices(patch))

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_9edfc990(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = objects(I, T, F, F)
    x1 = colorfilter(x0, ZERO)
    x2 = ofcolor(I, ONE)
    x3 = rbind(adjacent, x2)
    x4 = mfilter(x1, x3)
    x5 = recolor(ONE, x4)
    x6 = paint(I, x5)
    return x6


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_9edfc990(inp)
        assert pred == _to_grid(expected), f"{name} failed"

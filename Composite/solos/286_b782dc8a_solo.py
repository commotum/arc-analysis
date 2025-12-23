# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "b782dc8a"
SERIAL = "286"
URL    = "https://arcprize.org/play?task=b782dc8a"

# --- Code Golf Concepts ---
CONCEPTS = [
    "pattern_expansion",
    "maze",
]

# --- Example Grids ---
E1_IN = np.array([
    [8, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 0, 8, 8, 8, 0, 8, 8, 0, 8, 8, 8, 0],
    [0, 0, 8, 8, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 8, 0, 8, 0, 0, 8, 0, 8, 0],
    [8, 8, 8, 0, 8, 0, 8, 8, 8, 8, 0, 8, 8, 8, 0, 8, 0, 8, 8, 8, 8, 0, 8, 0],
    [8, 0, 0, 0, 8, 0, 8, 0, 0, 8, 0, 0, 0, 8, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0],
    [8, 0, 8, 8, 8, 0, 8, 8, 0, 8, 0, 8, 8, 8, 0, 8, 8, 0, 8, 8, 8, 8, 8, 0],
    [8, 0, 8, 0, 0, 0, 0, 8, 0, 8, 0, 8, 0, 0, 0, 0, 8, 0, 8, 0, 0, 0, 0, 0],
    [8, 0, 8, 8, 8, 8, 8, 8, 0, 8, 0, 8, 8, 8, 8, 8, 8, 3, 8, 8, 8, 8, 8, 0],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 3, 2, 3, 0, 0, 0, 8, 0],
    [8, 8, 0, 8, 8, 8, 0, 8, 8, 8, 0, 8, 8, 8, 8, 8, 8, 3, 8, 8, 8, 0, 8, 0],
    [0, 8, 0, 8, 0, 8, 0, 8, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 8, 0, 8, 0, 8, 0],
    [0, 8, 8, 8, 0, 8, 8, 8, 0, 8, 8, 8, 0, 8, 8, 0, 8, 8, 8, 0, 8, 8, 8, 0],
], dtype=int)

E1_OUT = np.array([
    [8, 3, 2, 3, 2, 3, 8, 8, 8, 8, 8, 8, 0, 8, 8, 8, 2, 8, 8, 0, 8, 8, 8, 0],
    [3, 2, 8, 8, 8, 2, 3, 2, 3, 2, 3, 8, 0, 0, 0, 8, 3, 8, 0, 0, 8, 2, 8, 0],
    [8, 8, 8, 0, 8, 3, 8, 8, 8, 8, 2, 8, 8, 8, 0, 8, 2, 8, 8, 8, 8, 3, 8, 0],
    [8, 0, 0, 0, 8, 2, 8, 0, 0, 8, 3, 2, 3, 8, 0, 8, 3, 2, 3, 2, 3, 2, 8, 0],
    [8, 0, 8, 8, 8, 3, 8, 8, 0, 8, 2, 8, 8, 8, 0, 8, 8, 3, 8, 8, 8, 8, 8, 0],
    [8, 0, 8, 2, 3, 2, 3, 8, 0, 8, 3, 8, 0, 0, 0, 0, 8, 2, 8, 0, 0, 0, 0, 0],
    [8, 0, 8, 8, 8, 8, 8, 8, 0, 8, 2, 8, 8, 8, 8, 8, 8, 3, 8, 8, 8, 8, 8, 0],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 8, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 8, 0],
    [8, 8, 0, 8, 8, 8, 0, 8, 8, 8, 2, 8, 8, 8, 8, 8, 8, 3, 8, 8, 8, 3, 8, 0],
    [0, 8, 0, 8, 0, 8, 0, 8, 3, 2, 3, 8, 0, 0, 0, 0, 8, 2, 8, 0, 8, 2, 8, 0],
    [0, 8, 8, 8, 0, 8, 8, 8, 2, 8, 8, 8, 0, 8, 8, 0, 8, 8, 8, 0, 8, 8, 8, 0],
], dtype=int)

E2_IN = np.array([
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 0, 0, 8],
    [8, 8, 0, 8, 8, 8, 0, 8, 0, 8, 8, 8, 0, 8],
    [0, 8, 0, 0, 0, 8, 0, 8, 0, 8, 0, 8, 8, 8],
    [0, 8, 8, 8, 8, 8, 0, 8, 0, 8, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 8, 0, 8, 8, 8, 0, 8],
    [8, 8, 8, 8, 8, 8, 0, 8, 0, 0, 0, 8, 0, 8],
    [8, 0, 0, 0, 0, 8, 0, 8, 8, 8, 0, 8, 0, 8],
    [8, 8, 8, 8, 0, 8, 0, 0, 0, 8, 0, 8, 0, 0],
    [0, 0, 0, 8, 1, 8, 8, 8, 8, 8, 0, 8, 8, 0],
    [8, 8, 0, 8, 4, 1, 0, 0, 0, 0, 0, 0, 8, 0],
    [0, 8, 0, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 0],
    [0, 8, 8, 8, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 8, 0, 8, 8, 8, 8, 8, 8, 8],
], dtype=int)

E2_OUT = np.array([
    [0, 0, 0, 8, 0, 0, 0, 8, 1, 4, 1, 4, 1, 8],
    [8, 8, 0, 8, 8, 8, 0, 8, 4, 8, 8, 8, 4, 8],
    [0, 8, 0, 0, 0, 8, 0, 8, 1, 8, 0, 8, 8, 8],
    [0, 8, 8, 8, 8, 8, 0, 8, 4, 8, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 8, 1, 8, 8, 8, 0, 8],
    [8, 8, 8, 8, 8, 8, 0, 8, 4, 1, 4, 8, 0, 8],
    [8, 4, 1, 4, 1, 8, 0, 8, 8, 8, 1, 8, 0, 8],
    [8, 8, 8, 8, 4, 8, 0, 0, 0, 8, 4, 8, 0, 0],
    [0, 0, 0, 8, 1, 8, 8, 8, 8, 8, 1, 8, 8, 0],
    [8, 8, 0, 8, 4, 1, 4, 1, 4, 1, 4, 1, 8, 0],
    [1, 8, 0, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 0],
    [4, 8, 8, 8, 4, 8, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 4, 1, 4, 1, 8, 0, 8, 8, 8, 8, 8, 8, 8],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [8, 8, 0, 8, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 8, 0, 8, 8, 8, 8, 4, 8, 8, 8, 8, 8, 8, 8],
    [0, 8, 0, 0, 0, 0, 4, 3, 8, 0, 0, 0, 0, 0, 8],
    [0, 8, 8, 8, 8, 8, 8, 4, 8, 8, 8, 0, 8, 8, 8],
    [0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 8, 0, 8, 0, 0],
    [8, 8, 8, 8, 8, 0, 8, 8, 8, 0, 8, 0, 8, 0, 8],
    [0, 0, 0, 0, 8, 0, 0, 0, 8, 0, 8, 0, 8, 0, 8],
    [8, 8, 8, 0, 8, 8, 8, 0, 8, 0, 8, 0, 8, 8, 8],
    [0, 0, 8, 0, 0, 0, 8, 0, 8, 0, 8, 0, 0, 0, 0],
    [8, 0, 8, 8, 8, 0, 8, 8, 8, 0, 8, 8, 8, 0, 8],
    [8, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 8],
    [8, 8, 8, 0, 8, 0, 8, 8, 8, 8, 8, 8, 8, 0, 8],
    [0, 0, 8, 0, 8, 0, 8, 0, 0, 0, 0, 0, 0, 0, 8],
    [8, 0, 8, 8, 8, 0, 8, 0, 8, 8, 8, 8, 8, 8, 8],
    [8, 0, 0, 0, 0, 0, 8, 0, 8, 0, 0, 0, 0, 0, 0],
], dtype=int)

T_OUT = np.array([
    [8, 8, 4, 8, 0, 0, 8, 3, 4, 3, 4, 3, 4, 3, 4],
    [0, 8, 3, 8, 8, 8, 8, 4, 8, 8, 8, 8, 8, 8, 8],
    [0, 8, 4, 3, 4, 3, 4, 3, 8, 0, 0, 0, 0, 0, 8],
    [0, 8, 8, 8, 8, 8, 8, 4, 8, 8, 8, 0, 8, 8, 8],
    [0, 0, 0, 0, 0, 0, 8, 3, 4, 3, 8, 0, 8, 0, 0],
    [8, 8, 8, 8, 8, 0, 8, 8, 8, 4, 8, 0, 8, 0, 8],
    [4, 3, 4, 3, 8, 0, 0, 0, 8, 3, 8, 0, 8, 0, 8],
    [8, 8, 8, 4, 8, 8, 8, 0, 8, 4, 8, 0, 8, 8, 8],
    [0, 0, 8, 3, 4, 3, 8, 0, 8, 3, 8, 0, 0, 0, 0],
    [8, 0, 8, 8, 8, 4, 8, 8, 8, 4, 8, 8, 8, 0, 8],
    [8, 0, 0, 0, 8, 3, 4, 3, 4, 3, 4, 3, 8, 0, 8],
    [8, 8, 8, 0, 8, 4, 8, 8, 8, 8, 8, 8, 8, 0, 8],
    [4, 3, 8, 0, 8, 3, 8, 0, 0, 0, 0, 0, 0, 0, 8],
    [8, 4, 8, 8, 8, 4, 8, 0, 8, 8, 8, 8, 8, 8, 8],
    [8, 3, 4, 3, 4, 3, 8, 0, 8, 0, 0, 0, 0, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
L=len
R=range
def p(g):
 h,w=L(g),L(g[0])
 f=sum(g,[]);C=sorted([[f.count(k),k] for k in set(f)])[:2]
 d={C[0][1]:C[1][1],C[1][1]:C[0][1]}
 for i in range(50):
  for r in R(h):
   for c in R(w):
    if g[r][c] in d:
     for y,x in [[0,1],[0,-1],[1,0],[-1,0]]:
      if 0<=r+y<h and 0<=c+x<w and g[r+y][c+x]==0:
       g[r+y][c+x]=d[g[r][c]]
 return g


# --- Code Golf Solution (Compressed) ---
def q(i, k=39):
    return -k * i or [[(t := (y or sum({*t % 8 * sum(i, x)} - {t, 8}))) for y in [8] + x][:0:-1] for *x, in zip(*p(i, k - 1))]


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

F = False

T = True

def identity(
    x: Any
) -> Any:
    """ identity function """
    return x

def even(
    n: Integer
) -> Boolean:
    """ evenness """
    return n % 2 == 0

def order(
    container: Container,
    compfunc: Callable
) -> Tuple:
    """ order container by custom key """
    return tuple(sorted(container, key=compfunc))

def size(
    container: Container
) -> Integer:
    """ cardinality """
    return len(container)

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

def ofcolor(
    grid: Grid,
    value: Integer
) -> Indices:
    """ indices of all grid cells with value """
    return frozenset((i, j) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value)

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

def generate_b782dc8a(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    wall_pairs = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
    dlt = [('W', (-1, 0)), ('E', (1, 0)), ('S', (0, 1)), ('N', (0, -1))]
    walls = {'N': True, 'S': True, 'E': True, 'W': True}
    fullsucc = False
    while True:
        h = unifint(diff_lb, diff_ub, (3, 15))
        w = unifint(diff_lb, diff_ub, (3, 15))
        maze = [[{'x': x, 'y': y, 'walls': {**walls}} for y in range(h)] for x in range(w)]
        kk = h * w
        stck = []
        cc = maze[0][0]
        nv = 1
        while nv < kk:
            nbhs = []
            for direc, (dx, dy) in dlt:
                x2, y2 = cc['x'] + dx, cc['y'] + dy
                if 0 <= x2 < w and 0 <= y2 < h:
                    neighbour = maze[x2][y2]
                    if all(neighbour['walls'].values()):
                        nbhs.append((direc, neighbour))
            if not nbhs:
                cc = stck.pop()
                continue
            direc, next_cell = choice(nbhs)
            cc['walls'][direc] = False
            next_cell['walls'][wall_pairs[direc]] = False
            stck.append(cc)
            cc = next_cell
            nv += 1
        pathcol, wallcol, dotcol, ncol = sample(cols, 4)
        grid = [[pathcol for x in range(w * 2)]]
        for y in range(h):
            row = [pathcol]
            for x in range(w):
                row.append(wallcol)
                row.append(pathcol if maze[x][y]['walls']['E'] else wallcol)
            grid.append(row)
            row = [pathcol]
            for x in range(w):
                row.append(pathcol if maze[x][y]['walls']['S'] else wallcol)
                row.append(pathcol)
            grid.append(row)
        gi = tuple(tuple(r[1:-1]) for r in grid[1:-1])
        objs = objects(gi, T, F, F)
        objs = colorfilter(objs, pathcol)
        objs = sfilter(objs, lambda obj: size(obj) > 4)
        if len(objs) == 0:
            continue
        objs = order(objs, size)
        nobjs = len(objs)
        idx = unifint(diff_lb, diff_ub, (0, nobjs - 1))
        obj = toindices(objs[idx])
        cell = choice(totuple(obj))
        gi = fill(gi, dotcol, {cell})
        nbhs = dneighbors(cell) & ofcolor(gi, pathcol)
        gi = fill(gi, ncol, nbhs)
        obj1 = sfilter(obj, lambda ij: even(manhattan({ij}, {cell})))
        obj2 = obj - obj1
        go = fill(gi, dotcol, obj1)
        go = fill(go, ncol, obj2)
        break
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
IntegerSet = FrozenSet[Integer]

ContainerContainer = Container[Container]

def combine(
    a: Container,
    b: Container
) -> Container:
    """ union """
    return type(a)((*a, *b))

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

def argmin(
    container: Container,
    compfunc: Callable
) -> Any:
    """ smallest item by custom order """
    return min(container, key=compfunc, default=None)

def initset(
    value: Any
) -> FrozenSet:
    """ initialize container """
    return frozenset({value})

def mfilter(
    container: Container,
    function: Callable
) -> FrozenSet:
    """ filter and merge """
    return merge(sfilter(container, function))

def first(
    container: Container
) -> Any:
    """ first item of container """
    return next(iter(container))

def remove(
    value: Any,
    container: Container
) -> Container:
    """ remove item from container """
    return type(container)(e for e in container if e != value)

def chain(
    h: Callable,
    g: Callable,
    f: Callable
) -> Callable:
    """ function composition with three functions """
    return lambda x: h(g(f(x)))

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

def mapply(
    function: Callable,
    container: ContainerContainer
) -> FrozenSet:
    """ apply and merge """
    return merge(apply(function, container))

def leastcolor(
    element: Element
) -> Integer:
    """ least common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return min(set(values), key=values.count)

def colorcount(
    element: Element,
    value: Integer
) -> Integer:
    """ number of cells with color """
    if isinstance(element, tuple):
        return sum(row.count(value) for row in element)
    return sum(v == value for v, _ in element)

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

def verify_b782dc8a(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = leastcolor(I)
    x1 = palette(I)
    x2 = remove(x0, x1)
    x3 = lbind(colorcount, I)
    x4 = argmin(x2, x3)
    x5 = ofcolor(I, x0)
    x6 = ofcolor(I, x4)
    x7 = combine(x5, x6)
    x8 = mapply(neighbors, x7)
    x9 = difference(x8, x7)
    x10 = toobject(x9, I)
    x11 = leastcolor(x10)
    x12 = ofcolor(I, x0)
    x13 = first(x12)
    x14 = initset(x13)
    x15 = objects(I, T, F, F)
    x16 = colorfilter(x15, x11)
    x17 = lbind(adjacent, x7)
    x18 = mfilter(x16, x17)
    x19 = toindices(x18)
    x20 = rbind(manhattan, x14)
    x21 = chain(even, x20, initset)
    x22 = sfilter(x19, x21)
    x23 = fill(I, x4, x19)
    x24 = fill(x23, x0, x22)
    return x24


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_b782dc8a(inp)
        assert pred == _to_grid(expected), f"{name} failed"

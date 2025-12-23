# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "6d0160f0"
SERIAL = "163"
URL    = "https://arcprize.org/play?task=6d0160f0"

# --- Code Golf Concepts ---
CONCEPTS = [
    "detect_grid",
    "separate_image",
    "find_the_intruder",
    "pattern_moving",
]

# --- Example Grids ---
E1_IN = np.array([
    [3, 0, 0, 5, 7, 0, 6, 5, 8, 0, 7],
    [0, 0, 9, 5, 0, 3, 0, 5, 0, 6, 0],
    [7, 2, 0, 5, 0, 0, 2, 5, 0, 3, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [7, 0, 2, 5, 8, 7, 0, 5, 0, 2, 3],
    [0, 0, 6, 5, 0, 0, 3, 5, 0, 0, 7],
    [3, 0, 0, 5, 2, 0, 0, 5, 0, 6, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 3, 4, 5, 0, 2, 0, 5, 2, 0, 7],
    [7, 0, 0, 5, 7, 0, 3, 5, 0, 0, 1],
    [0, 0, 2, 5, 0, 6, 0, 5, 0, 3, 0],
], dtype=int)

E1_OUT = np.array([
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 3, 4],
    [0, 0, 0, 5, 0, 0, 0, 5, 7, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 2],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
], dtype=int)

E2_IN = np.array([
    [3, 0, 0, 5, 0, 2, 0, 5, 0, 6, 0],
    [0, 0, 7, 5, 0, 0, 0, 5, 0, 0, 9],
    [0, 6, 0, 5, 0, 1, 0, 5, 0, 0, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 3, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [1, 0, 9, 5, 0, 0, 6, 5, 0, 7, 3],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [9, 0, 0, 5, 0, 9, 0, 5, 0, 9, 0],
    [0, 6, 0, 5, 0, 0, 4, 5, 0, 0, 1],
    [0, 0, 0, 5, 7, 0, 0, 5, 0, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 9, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 4],
    [0, 0, 0, 5, 0, 0, 0, 5, 7, 0, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
], dtype=int)

E3_IN = np.array([
    [0, 7, 0, 5, 0, 6, 0, 5, 7, 0, 0],
    [8, 3, 6, 5, 0, 0, 0, 5, 0, 8, 0],
    [0, 0, 0, 5, 0, 3, 0, 5, 0, 0, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 8, 7, 5, 0, 3, 0, 5, 0, 0, 7],
    [0, 0, 0, 5, 8, 0, 0, 5, 0, 8, 6],
    [0, 0, 6, 5, 0, 0, 0, 5, 3, 0, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 6, 0, 5, 0, 8, 0, 5, 0, 0, 0],
    [8, 0, 0, 5, 3, 0, 0, 5, 4, 0, 8],
    [0, 7, 0, 5, 0, 6, 0, 5, 0, 6, 7],
], dtype=int)

E3_OUT = np.array([
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [4, 0, 8, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 6, 7, 5, 0, 0, 0, 5, 0, 0, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
], dtype=int)

E4_IN = np.array([
    [3, 0, 0, 5, 0, 1, 0, 5, 0, 0, 2],
    [0, 2, 0, 5, 0, 3, 0, 5, 0, 6, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 1, 0, 5, 0, 0, 0, 5, 0, 3, 0],
    [7, 0, 6, 5, 2, 0, 7, 5, 0, 7, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 6, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [7, 0, 0, 5, 0, 4, 0, 5, 0, 0, 3],
    [0, 0, 0, 5, 0, 7, 0, 5, 2, 0, 0],
    [0, 3, 0, 5, 0, 3, 0, 5, 0, 0, 6],
], dtype=int)

E4_OUT = np.array([
    [0, 0, 0, 5, 0, 4, 0, 5, 0, 0, 0],
    [0, 0, 0, 5, 0, 7, 0, 5, 0, 0, 0],
    [0, 0, 0, 5, 0, 3, 0, 5, 0, 0, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [2, 0, 3, 5, 2, 0, 0, 5, 0, 3, 0],
    [7, 6, 0, 5, 0, 7, 0, 5, 6, 7, 0],
    [0, 0, 0, 5, 6, 0, 3, 5, 0, 0, 2],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [7, 0, 0, 5, 0, 0, 0, 5, 6, 0, 4],
    [0, 6, 0, 5, 0, 2, 7, 5, 0, 2, 0],
    [6, 0, 2, 5, 0, 3, 0, 5, 0, 7, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [7, 0, 0, 5, 0, 6, 0, 5, 2, 3, 0],
    [0, 0, 6, 5, 0, 2, 0, 5, 0, 0, 0],
    [2, 0, 0, 5, 0, 7, 0, 5, 0, 6, 7],
], dtype=int)

T_OUT = np.array([
    [0, 0, 0, 5, 0, 0, 0, 5, 6, 0, 4],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 2, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 7, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(g):
 R=range
 for r in R(3):
  for c in R(3):
   b=[[g[4*r+i][4*c+j]for j in R(3)]for i in R(3)]
   for i in R(3):
    for j in R(3):
     if b[i][j]==4:
      z=[[0]*11for _ in R(11)]
      for x in R(3):
       for y in R(3):z[4*i+x][4*j+y]=b[x][y]
      for k in R(11):z[k][3]=z[k][7]=z[3][k]=z[7][k]=5
      return z


# --- Code Golf Solution (Compressed) ---
def q(g):
    return [[max((5 * (g[y][x] == 5) or (g[p % 3 * 4 + y // 4][p & -4 | x // 4] == 4) * g[p % 3 * 4 + y % 4][p & -4 | x % 4] for p in R)) for x in R] for y in R]


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, sample, uniform

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Numerical = Union[Integer, IntegerTuple]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

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

def difference(
    a: Container,
    b: Container
) -> Container:
    """ difference """
    return type(a)(e for e in a if e not in b)

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

def generate_6d0160f0(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (4,))
    h = unifint(diff_lb, diff_ub, (2, 5))
    w = unifint(diff_lb, diff_ub, (2, 5))
    nh, nw = h, w
    bgc, linc = sample(cols, 2)
    fullh = h * nh + nh - 1
    fullw = w * nw + nw - 1
    gi = canvas(bgc, (fullh, fullw))
    for iloc in range(h, fullh, h+1):
        gi = fill(gi, linc, hfrontier((iloc, 0)))
    for jloc in range(w, fullw, w+1):
        gi = fill(gi, linc, vfrontier((0, jloc)))
    noccs = unifint(diff_lb, diff_ub, (1, h * w))
    denseinds = asindices(canvas(-1, (h, w)))
    sparseinds = {(a*(h+1), b*(w+1)) for a, b in denseinds}
    locs = sample(totuple(sparseinds), noccs)
    trgtl = choice(locs)
    remlocs = remove(trgtl, locs)
    ntrgt = unifint(diff_lb, diff_ub, (1, (h * w - 1)))
    place = choice(totuple(denseinds))
    ncols = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(cols, ncols)
    candss = totuple(remove(place, denseinds))
    trgrem = sample(candss, ntrgt)
    trgrem = {(choice(ccols), ij) for ij in trgrem}
    trgtobj = {(4, place)} | trgrem
    go = paint(gi, shift(sfilter(trgtobj, lambda cij: cij[0] != linc), multiply(place, increment((h, w)))))
    gi = paint(gi, shift(trgtobj, trgtl))
    toleaveout = ccols
    for rl in remlocs:
        tlo = choice(totuple(ccols))
        ncells = unifint(diff_lb, diff_ub, (1, h * w - 1))
        inds = sample(totuple(denseinds), ncells)
        obj = {(choice(remove(tlo, ccols) if len(ccols) > 1 else ccols), ij) for ij in inds}
        toleaveout = remove(tlo, toleaveout)
        gi = paint(gi, shift(obj, rl))
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

IntegerSet = FrozenSet[Integer]

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

ContainerContainer = Container[Container]

FOUR = 4

F = False

T = True

NEG_ONE = -1

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

def size(
    container: Container
) -> Integer:
    """ cardinality """
    return len(container)

def merge(
    containers: ContainerContainer
) -> Container:
    """ merging """
    return type(containers)(e for c in containers for e in c)

def valmax(
    container: Container,
    compfunc: Callable
) -> Integer:
    """ maximum by custom function """
    return compfunc(max(container, key=compfunc, default=0))

def argmax(
    container: Container,
    compfunc: Callable
) -> Any:
    """ largest item by custom order """
    return max(container, key=compfunc, default=None)

def extract(
    container: Container,
    condition: Callable
) -> Any:
    """ first element of container that satisfies condition """
    return next(e for e in container if condition(e))

def first(
    container: Container
) -> Any:
    """ first item of container """
    return next(iter(container))

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

def colorcount(
    element: Element,
    value: Integer
) -> Integer:
    """ number of cells with color """
    if isinstance(element, tuple):
        return sum(row.count(value) for row in element)
    return sum(v == value for v, _ in element)

def ulcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of upper left corner """
    return tuple(map(min, zip(*toindices(patch))))

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

def palette(
    element: Element
) -> IntegerSet:
    """ colors occurring in object or grid """
    if isinstance(element, tuple):
        return frozenset({v for r in element for v in r})
    return frozenset({v for v, _ in element})

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

def hconcat(
    a: Grid,
    b: Grid
) -> Grid:
    """ concatenate two grids horizontally """
    return tuple(i + j for i, j in zip(a, b))

def center(
    patch: Patch
) -> IntegerTuple:
    """ center of the patch """
    return (uppermost(patch) + height(patch) // 2, leftmost(patch) + width(patch) // 2)

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

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_6d0160f0(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = frontiers(I)
    x1 = merge(x0)
    x2 = mostcolor(x1)
    x3 = shape(I)
    x4 = canvas(NEG_ONE, x3)
    x5 = hconcat(I, x4)
    x6 = fill(x5, NEG_ONE, x1)
    x7 = objects(x6, F, F, T)
    x8 = lbind(contained, FOUR)
    x9 = compose(x8, palette)
    x10 = extract(x7, x9)
    x11 = lbind(sfilter, x7)
    x12 = compose(size, x11)
    x13 = rbind(compose, palette)
    x14 = lbind(lbind, contained)
    x15 = chain(x12, x13, x14)
    x16 = merge(x7)
    x17 = palette(I)
    x18 = remove(x2, x17)
    x19 = valmax(x18, x15)
    x20 = matcher(x15, x19)
    x21 = sfilter(x18, x20)
    x22 = lbind(colorcount, x16)
    x23 = argmax(x21, x22)
    x24 = shape(I)
    x25 = canvas(x23, x24)
    x26 = paint(x25, x1)
    x27 = normalize(x10)
    x28 = matcher(first, x2)
    x29 = compose(flip, x28)
    x30 = sfilter(x27, x29)
    x31 = shape(x27)
    x32 = increment(x31)
    x33 = matcher(first, FOUR)
    x34 = sfilter(x27, x33)
    x35 = center(x34)
    x36 = multiply(x32, x35)
    x37 = shift(x30, x36)
    x38 = paint(x26, x37)
    return x38


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("E4", E4_IN, E4_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_6d0160f0(inp)
        assert pred == _to_grid(expected), f"{name} failed"

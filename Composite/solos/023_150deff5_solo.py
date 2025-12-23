# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "150deff5"
SERIAL = "023"
URL    = "https://arcprize.org/play?task=150deff5"

# --- Code Golf Concepts ---
CONCEPTS = [
    "pattern_coloring",
    "pattern_deconstruction",
    "associate_colors_to_patterns",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 5, 5, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 5, 5, 5, 5, 5, 0, 0, 0, 0],
    [0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0],
    [0, 0, 0, 5, 5, 5, 5, 5, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 5, 5, 0, 0, 0],
    [0, 0, 0, 0, 0, 5, 5, 5, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 8, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 8, 8, 2, 2, 2, 0, 0, 0, 0],
    [0, 0, 0, 2, 8, 8, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 8, 8, 8, 8, 0, 0, 0],
    [0, 0, 0, 2, 0, 0, 8, 8, 0, 0, 0],
    [0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E2_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 5, 5, 5, 5, 5, 5, 0, 0, 0],
    [0, 5, 5, 5, 5, 5, 5, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 5, 0, 0, 0],
    [0, 0, 0, 0, 5, 5, 5, 0, 0, 0],
    [0, 0, 0, 0, 5, 5, 5, 0, 0, 0],
    [0, 0, 0, 0, 5, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 8, 8, 2, 8, 8, 2, 0, 0, 0],
    [0, 8, 8, 2, 8, 8, 2, 0, 0, 0],
    [0, 0, 0, 2, 0, 0, 2, 0, 0, 0],
    [0, 0, 0, 0, 2, 8, 8, 0, 0, 0],
    [0, 0, 0, 0, 2, 8, 8, 0, 0, 0],
    [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E3_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 5, 5, 5, 5, 5, 0, 0, 0],
    [0, 0, 0, 0, 5, 5, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 0, 0],
    [0, 0, 0, 5, 5, 5, 0, 0, 0],
    [0, 0, 0, 5, 5, 5, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 2, 2, 8, 8, 0, 0, 0],
    [0, 0, 0, 0, 8, 8, 0, 0, 0],
    [0, 0, 0, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 8, 8, 0, 0, 0],
    [0, 0, 0, 2, 8, 8, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 5, 5, 0, 5, 5, 5, 0, 0, 0],
    [0, 0, 5, 5, 0, 0, 5, 0, 0, 0, 0],
    [0, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0],
    [0, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0],
    [0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 5, 5, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 5, 5, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

T_OUT = np.array([
    [0, 0, 8, 8, 0, 2, 2, 2, 0, 0, 0],
    [0, 0, 8, 8, 0, 0, 2, 0, 0, 0, 0],
    [0, 2, 2, 2, 8, 8, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 8, 2, 0, 0, 0, 0],
    [0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 8, 8, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 8, 8, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(g,L=len,R=range):
 #rules: 1x3/3x1 for all reds, 2x2 for all blues, no gray remaining
 h,w=L(g),L(g[0])
 Z=[[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]] #3x3
 P=[[0,0],[0,1],[1,0],[1,1]] #2x2
 Q=[[0,0],[0,1],[0,2]] #1x3
 S=[[0,0],[1,0],[2,0]] #3x1
 for r in R(h):
  for c in R(w):
   try:
    if [g[r+i[0]][c+i[1]] for i in Z]==[5,5,5,5,5,5,0,0,5]:
     Y=[8,8,2,8,8,2,0,0,2]
     for i in R(L(Z)): 
      g[r+Z[i][0]][c+Z[i][1]]=Y[i]
    elif [g[r+i[0]][c+i[1]] for i in Z]==[5,5,5,5,5,5,5,0,0]:
     Y=[2,8,8,2,8,8,2,0,0]
     for i in R(L(Z)): 
      g[r+Z[i][0]][c+Z[i][1]]=Y[i]
    elif [g[r+i[0]][c+i[1]] for i in Z]==[0,5,5,0,5,5,5,5,5]:
     Y=[0,8,8,0,8,8,2,2,2]
     for i in R(L(Z)): 
      g[r+Z[i][0]][c+Z[i][1]]=Y[i]
    elif [g[r+i[0]][c+i[1]] for i in Z]==[5,5,5,5,5,0,5,5,0]:
     Y=[2,2,2,8,8,0,8,8,0]
     for i in R(L(Z)): 
      g[r+Z[i][0]][c+Z[i][1]]=Y[i]
   except: pass
 for r in R(h):
  for c in R(w):
   try:
    if [g[r+i[0]][c+i[1]] for i in P]==[5,5,5,5]:
     for i in P: 
      g[r+i[0]][c+i[1]]=8
    elif [g[r+i[0]][c+i[1]] for i in Q]==[5,5,5]:
     for i in Q: 
      g[r+i[0]][c+i[1]]=2
    elif [g[r+i[0]][c+i[1]] for i in S]==[5,5,5]:
     for i in S: 
      g[r+i[0]][c+i[1]]=2
   except: pass
 return g


# --- Code Golf Solution (Compressed) ---
def q(i, w=2):
    return s != (r := re.sub((w % 2 * '5, ' + '5(.%s)??') % {w * 3 % -7 % len(i[0] * 3) + 2} * (3 - w % 2), ' 82,\\81\\ 12 \\82, 82'[w::2], s, 1)) and p(eval(r)) or (w and p(i, w - 1)) if '5' in (s := str(i)) else i


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

Piece = Union[Grid, Patch]

def add(
    a: Numerical,
    b: Numerical
) -> Numerical:
    """ addition """
    if isinstance(a, int) and isinstance(b, int):
        return a + b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] + b[0], a[1] + b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a + b[0], a + b[1])
    return (a[0] + b, a[1] + b)

def difference(
    a: Container,
    b: Container
) -> Container:
    """ difference """
    return type(a)(e for e in a if e not in b)

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

def generate_150deff5(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (2, 8))
    bo = {(0, 0), (0, 1), (1, 0), (1, 1)}
    ro1 = {(0, 0), (0, 1), (0, 2)}
    ro2 = {(0, 0), (1, 0), (2, 0)}
    boforb = set()
    reforb = set()
    h = unifint(diff_lb, diff_ub, (8, 30))
    w = unifint(diff_lb, diff_ub, (8, 30))
    bgc, fgc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    noccs = unifint(diff_lb, diff_ub, (2, (h * w) // 10))
    inds = asindices(gi)
    needsbgc = []
    for k in range(noccs):
        obj, col = choice(((bo, 8), (choice((ro1, ro2)), 2)))
        oh, ow = shape(obj)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow and shift(obj, ij).issubset(inds))
        if col == 8:
            cands = sfilter(cands, lambda ij: ij not in boforb)
        else:
            cands = sfilter(cands, lambda ij: ij not in reforb)
        if len(cands) == 0:
            break
        loc = choice(totuple(cands))
        if col == 8:
            boforb.add(add(loc, (-2, 0)))
            boforb.add(add(loc, (2, 0)))
            boforb.add(add(loc, (0, 2)))
            boforb.add(add(loc, (0, -2)))
        if col == 2:
            if obj == ro1:
                reforb.add(add(loc, (0, 3)))
                reforb.add(add(loc, (0, -3)))
            else:
                reforb.add(add(loc, (1, 0)))
                reforb.add(add(loc, (-1, 0)))
        plcd = shift(obj, loc)
        gi = fill(gi, fgc, plcd)
        go = fill(go, col, plcd)
        inds = inds - plcd
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Element = Union[Object, Grid]

ContainerContainer = Container[Container]

ONE = 1

TWO = 2

THREE = 3

FOUR = 4

FIVE = 5

EIGHT = 8

ORIGIN = (0, 0)

UNITY = (1, 1)

TWO_BY_TWO = (2, 2)

THREE_BY_THREE = (3, 3)

def identity(
    x: Any
) -> Any:
    """ identity function """
    return x

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

def initset(
    value: Any
) -> FrozenSet:
    """ initialize container """
    return frozenset({value})

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

def insert(
    value: Any,
    container: FrozenSet
) -> FrozenSet:
    """ insert item into container """
    return container.union(frozenset({value}))

def remove(
    value: Any,
    container: Container
) -> Container:
    """ remove item from container """
    return type(container)(e for e in container if e != value)

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

def power(
    function: Callable,
    n: Integer
) -> Callable:
    """ power of function """
    if n == 1:
        return function
    return compose(function, power(function, n - 1))

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

def recolor(
    value: Integer,
    patch: Patch
) -> Object:
    """ recolor patch """
    return frozenset((value, index) for index in toindices(patch))

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

def asobject(
    grid: Grid
) -> Object:
    """ conversion of grid to object """
    return frozenset((v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r))

def rot90(
    grid: Grid
) -> Grid:
    """ quarter clockwise rotation """
    return tuple(row for row in zip(*grid[::-1]))

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

def trim(
    grid: Grid
) -> Grid:
    """ trim border of grid """
    return tuple(r[1:-1] for r in grid[1:-1])

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

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_150deff5(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = mostcolor(I)
    x1 = leastcolor(I)
    x2 = shape(I)
    x3 = add(TWO, x2)
    x4 = canvas(x0, x3)
    x5 = asobject(I)
    x6 = shift(x5, UNITY)
    x7 = paint(x4, x6)
    x8 = astuple(TWO, ONE)
    x9 = dneighbors(UNITY)
    x10 = remove(x8, x9)
    x11 = recolor(x0, x10)
    x12 = initset(UNITY)
    x13 = recolor(x1, x12)
    x14 = combine(x11, x13)
    x15 = astuple(THREE, ONE)
    x16 = connect(UNITY, x15)
    x17 = recolor(TWO, x16)
    x18 = initset(TWO_BY_TWO)
    x19 = insert(UNITY, x18)
    x20 = backdrop(x19)
    x21 = astuple(TWO, THREE)
    x22 = astuple(THREE, TWO)
    x23 = initset(x22)
    x24 = insert(x21, x23)
    x25 = insert(THREE_BY_THREE, x24)
    x26 = recolor(x1, x20)
    x27 = outbox(x20)
    x28 = difference(x27, x25)
    x29 = recolor(x0, x28)
    x30 = combine(x26, x29)
    x31 = recolor(EIGHT, x20)
    x32 = lbind(lbind, shift)
    x33 = compose(x32, last)
    x34 = lbind(fork, paint)
    x35 = lbind(x34, identity)
    x36 = lbind(lbind, mapply)
    x37 = compose(x36, x33)
    x38 = lbind(rbind, occurrences)
    x39 = compose(x38, first)
    x40 = fork(compose, x37, x39)
    x41 = compose(x35, x40)
    x42 = astuple(x14, x17)
    x43 = x41(x42)
    x44 = compose(rot90, x43)
    x45 = power(x44, FOUR)
    x46 = astuple(x30, x31)
    x47 = x41(x46)
    x48 = compose(rot90, x47)
    x49 = power(x48, FOUR)
    x50 = compose(x45, x49)
    x51 = initset(ORIGIN)
    x52 = difference(x51, x51)
    x53 = lbind(recolor, TWO)
    x54 = rbind(ofcolor, TWO)
    x55 = compose(x53, x54)
    x56 = lbind(recolor, EIGHT)
    x57 = rbind(ofcolor, EIGHT)
    x58 = compose(x56, x57)
    x59 = fork(combine, x55, x58)
    x60 = lbind(recolor, x0)
    x61 = compose(x60, x59)
    x62 = fork(paint, identity, x61)
    x63 = chain(x62, x50, first)
    x64 = chain(x59, x50, first)
    x65 = fork(combine, last, x64)
    x66 = fork(astuple, x63, x65)
    x67 = astuple(x7, x52)
    x68 = power(x66, FIVE)
    x69 = x68(x67)
    x70 = first(x69)
    x71 = last(x69)
    x72 = paint(x70, x71)
    x73 = trim(x72)
    return x73


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_150deff5(inp)
        assert pred == _to_grid(expected), f"{name} failed"

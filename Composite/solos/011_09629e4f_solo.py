# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "09629e4f"
SERIAL = "011"
URL    = "https://arcprize.org/play?task=09629e4f"

# --- Code Golf Concepts ---
CONCEPTS = [
    "detect_grid",
    "separate_images",
    "count_tiles",
    "take_minimum",
    "enlarge_image",
    "create_grid",
    "adapt_image_to_grid",
]

# --- Example Grids ---
E1_IN = np.array([
    [2, 0, 0, 5, 0, 6, 2, 5, 0, 0, 4],
    [0, 4, 3, 5, 4, 0, 8, 5, 3, 0, 6],
    [6, 0, 0, 5, 3, 0, 0, 5, 8, 0, 2],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [3, 8, 0, 5, 6, 2, 0, 5, 0, 4, 8],
    [0, 0, 4, 5, 0, 0, 4, 5, 6, 0, 0],
    [6, 2, 0, 5, 3, 8, 0, 5, 0, 3, 2],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 3, 6, 5, 0, 2, 0, 5, 0, 6, 0],
    [2, 0, 0, 5, 4, 0, 8, 5, 0, 0, 8],
    [8, 0, 4, 5, 6, 3, 0, 5, 2, 3, 4],
], dtype=int)

E1_OUT = np.array([
    [2, 2, 2, 5, 0, 0, 0, 5, 0, 0, 0],
    [2, 2, 2, 5, 0, 0, 0, 5, 0, 0, 0],
    [2, 2, 2, 5, 0, 0, 0, 5, 0, 0, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 5, 4, 4, 4, 5, 3, 3, 3],
    [0, 0, 0, 5, 4, 4, 4, 5, 3, 3, 3],
    [0, 0, 0, 5, 4, 4, 4, 5, 3, 3, 3],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [6, 6, 6, 5, 0, 0, 0, 5, 0, 0, 0],
    [6, 6, 6, 5, 0, 0, 0, 5, 0, 0, 0],
    [6, 6, 6, 5, 0, 0, 0, 5, 0, 0, 0],
], dtype=int)

E2_IN = np.array([
    [2, 0, 3, 5, 4, 6, 0, 5, 0, 6, 0],
    [0, 0, 8, 5, 0, 0, 2, 5, 4, 0, 3],
    [4, 6, 0, 5, 3, 8, 0, 5, 2, 0, 8],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [4, 0, 8, 5, 0, 0, 2, 5, 0, 6, 4],
    [0, 0, 2, 5, 0, 3, 0, 5, 3, 0, 0],
    [3, 0, 6, 5, 4, 0, 6, 5, 8, 0, 2],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [3, 6, 0, 5, 0, 8, 4, 5, 2, 0, 0],
    [0, 8, 4, 5, 2, 0, 0, 5, 8, 0, 3],
    [2, 0, 0, 5, 0, 3, 6, 5, 6, 4, 0],
], dtype=int)

E2_OUT = np.array([
    [0, 0, 0, 5, 0, 0, 0, 5, 2, 2, 2],
    [0, 0, 0, 5, 0, 0, 0, 5, 2, 2, 2],
    [0, 0, 0, 5, 0, 0, 0, 5, 2, 2, 2],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 5, 3, 3, 3, 5, 0, 0, 0],
    [0, 0, 0, 5, 3, 3, 3, 5, 0, 0, 0],
    [0, 0, 0, 5, 3, 3, 3, 5, 0, 0, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [4, 4, 4, 5, 0, 0, 0, 5, 6, 6, 6],
    [4, 4, 4, 5, 0, 0, 0, 5, 6, 6, 6],
    [4, 4, 4, 5, 0, 0, 0, 5, 6, 6, 6],
], dtype=int)

E3_IN = np.array([
    [0, 3, 0, 5, 0, 6, 3, 5, 0, 6, 2],
    [6, 0, 4, 5, 2, 8, 0, 5, 0, 0, 8],
    [0, 2, 8, 5, 0, 4, 0, 5, 3, 0, 4],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 2, 0, 5, 4, 0, 3, 5, 3, 4, 0],
    [4, 0, 8, 5, 2, 0, 6, 5, 0, 0, 2],
    [3, 6, 0, 5, 0, 8, 0, 5, 8, 6, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [6, 3, 0, 5, 0, 3, 0, 5, 0, 0, 3],
    [0, 0, 2, 5, 0, 6, 4, 5, 2, 8, 0],
    [8, 4, 0, 5, 2, 0, 0, 5, 4, 0, 6],
], dtype=int)

E3_OUT = np.array([
    [0, 0, 0, 5, 3, 3, 3, 5, 0, 0, 0],
    [0, 0, 0, 5, 3, 3, 3, 5, 0, 0, 0],
    [0, 0, 0, 5, 3, 3, 3, 5, 0, 0, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 5, 6, 6, 6, 5, 4, 4, 4],
    [0, 0, 0, 5, 6, 6, 6, 5, 4, 4, 4],
    [0, 0, 0, 5, 6, 6, 6, 5, 4, 4, 4],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [2, 2, 2, 5, 0, 0, 0, 5, 0, 0, 0],
    [2, 2, 2, 5, 0, 0, 0, 5, 0, 0, 0],
    [2, 2, 2, 5, 0, 0, 0, 5, 0, 0, 0],
], dtype=int)

E4_IN = np.array([
    [3, 8, 4, 5, 4, 6, 0, 5, 2, 0, 8],
    [0, 0, 0, 5, 8, 0, 3, 5, 6, 0, 3],
    [6, 2, 0, 5, 0, 2, 0, 5, 4, 0, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 4, 2, 5, 8, 0, 3, 5, 0, 4, 0],
    [0, 8, 6, 5, 0, 0, 4, 5, 0, 2, 6],
    [0, 3, 0, 5, 2, 6, 0, 5, 0, 3, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 6, 0, 5, 6, 2, 0, 5, 3, 6, 0],
    [3, 0, 8, 5, 0, 8, 3, 5, 0, 0, 4],
    [4, 2, 0, 5, 0, 0, 4, 5, 2, 0, 8],
], dtype=int)

E4_OUT = np.array([
    [0, 0, 0, 5, 4, 4, 4, 5, 0, 0, 0],
    [0, 0, 0, 5, 4, 4, 4, 5, 0, 0, 0],
    [0, 0, 0, 5, 4, 4, 4, 5, 0, 0, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 5, 2, 2, 2, 5, 6, 6, 6],
    [0, 0, 0, 5, 2, 2, 2, 5, 6, 6, 6],
    [0, 0, 0, 5, 2, 2, 2, 5, 6, 6, 6],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 5, 3, 3, 3, 5, 0, 0, 0],
    [0, 0, 0, 5, 3, 3, 3, 5, 0, 0, 0],
    [0, 0, 0, 5, 3, 3, 3, 5, 0, 0, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [6, 4, 0, 5, 0, 3, 0, 5, 0, 4, 0],
    [0, 0, 3, 5, 2, 8, 6, 5, 8, 0, 2],
    [2, 0, 8, 5, 4, 0, 0, 5, 6, 3, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [2, 0, 0, 5, 0, 3, 0, 5, 3, 6, 2],
    [3, 4, 6, 5, 8, 4, 2, 5, 0, 0, 4],
    [0, 8, 0, 5, 0, 0, 6, 5, 8, 0, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 2, 4, 5, 0, 6, 4, 5, 0, 2, 8],
    [0, 6, 3, 5, 0, 0, 3, 5, 4, 0, 6],
    [0, 0, 0, 5, 2, 0, 8, 5, 3, 0, 0],
], dtype=int)

T_OUT = np.array([
    [0, 0, 0, 5, 2, 2, 2, 5, 4, 4, 4],
    [0, 0, 0, 5, 2, 2, 2, 5, 4, 4, 4],
    [0, 0, 0, 5, 2, 2, 2, 5, 4, 4, 4],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 5, 6, 6, 6, 5, 3, 3, 3],
    [0, 0, 0, 5, 6, 6, 6, 5, 3, 3, 3],
    [0, 0, 0, 5, 6, 6, 6, 5, 3, 3, 3],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
 A=range
 for c in A(3):
  for E in A(3):
   if sum(j[c*4+W][E*4+l]==0for W in A(3)for l in A(3))==5:
    k=[[5if i%4==3 or j%4==3else 0for j in A(11)]for i in A(11)]
    for W in A(3):
     for l in A(3):
      J=j[c*4+W][E*4+l]
      if J:
       for a in A(3):
        for C in A(3):k[W*4+a][l*4+C]=J
    return k


# --- Code Golf Solution (Compressed) ---
def q(*args, **kwargs):
    return (eval("lambda a:max(a*(not'8'in'%s'%a)" + f'for*a,in[*map(zip,a,a,a{',a[3:]*9,*[a[%d:]]*3' * 2 % (1, 2)})][::4]' * 2 + ')'))(*args, **kwargs)


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

def product(
    a: Container,
    b: Container
) -> FrozenSet:
    """ cartesian product """
    return frozenset((i, j) for j in b for i in a)

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

def generate_09629e4f(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 5))
    w = unifint(diff_lb, diff_ub, (2, 5))
    nrows, ncolumns = h, w
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    barcol = choice(remcols)
    remcols = remove(barcol, remcols)
    ncols = unifint(diff_lb, diff_ub, (2, min(7, (h * w) - 2)))
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    fullh, fullw = h * nrows + nrows - 1, w * ncolumns + ncolumns - 1
    gi = canvas(barcol, (fullh, fullw))
    locs = totuple(product(interval(0, fullh, h + 1), interval(0, fullw, w + 1)))
    trgloc = choice(locs)
    remlocs = remove(trgloc, locs)
    colssf = sample(remcols, ncols)
    colsss = remove(choice(colssf), colssf)
    trgssf = sample(inds, ncols - 1)
    gi = fill(gi, bgc, shift(inds, trgloc))
    for ij, cl in zip(trgssf, colsss):
        gi = fill(gi, cl, {add(trgloc, ij)})
    for rl in remlocs:
        trgss = sample(inds, ncols)
        tmpg = tuple(e for e in c)
        for ij, cl in zip(trgss, colssf):
            tmpg = fill(tmpg, cl, {ij})
        gi = paint(gi, shift(asobject(tmpg), rl))
    go = canvas(bgc, (fullh, fullw))
    go = fill(go, barcol, ofcolor(gi, barcol))
    for ij, cl in zip(trgssf, colsss):
        go = fill(go, cl, shift(inds, multiply(ij, (h+1, w+1))))
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

IntegerSet = FrozenSet[Integer]

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

ContainerContainer = Container[Container]

F = False

T = True

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

def last(
    container: Container
) -> Any:
    """ last item of container """
    return max(enumerate(container))[1]

def astuple(
    a: Integer,
    b: Integer
) -> IntegerTuple:
    """ constructs a tuple """
    return (a, b)

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

def ulcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of upper left corner """
    return tuple(map(min, zip(*toindices(patch))))

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

def vline(
    patch: Patch
) -> Boolean:
    """ whether the piece forms a vertical line """
    return height(patch) == len(patch) and width(patch) == 1

def hline(
    patch: Patch
) -> Boolean:
    """ whether the piece forms a horizontal line """
    return width(patch) == len(patch) and height(patch) == 1

def palette(
    element: Element
) -> IntegerSet:
    """ colors occurring in object or grid """
    if isinstance(element, tuple):
        return frozenset({v for r in element for v in r})
    return frozenset({v for v, _ in element})

def numcolors(
    element: Element
) -> IntegerSet:
    """ number of colors occurring in object or grid """
    return len(palette(element))

def color(
    obj: Object
) -> Integer:
    """ color of object """
    return next(iter(obj))[0]

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

def verify_09629e4f(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = frontiers(I)
    x1 = sfilter(x0, hline)
    x2 = sfilter(x0, vline)
    x3 = size(x1)
    x4 = size(x2)
    x5 = merge(x0)
    x6 = color(x5)
    x7 = shape(I)
    x8 = canvas(x6, x7)
    x9 = hconcat(I, x8)
    x10 = objects(x9, F, T, T)
    x11 = argmin(x10, numcolors)
    x12 = normalize(x11)
    x13 = toindices(x12)
    x14 = increment(x3)
    x15 = increment(x14)
    x16 = increment(x4)
    x17 = increment(x16)
    x18 = astuple(x15, x17)
    x19 = lbind(shift, x13)
    x20 = rbind(multiply, x18)
    x21 = chain(x19, x20, last)
    x22 = fork(recolor, first, x21)
    x23 = normalize(x11)
    x24 = mapply(x22, x23)
    x25 = paint(x8, x24)
    return x25


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("E4", E4_IN, E4_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_09629e4f(inp)
        assert pred == _to_grid(expected), f"{name} failed"

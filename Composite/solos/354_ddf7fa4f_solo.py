# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "ddf7fa4f"
SERIAL = "354"
URL    = "https://arcprize.org/play?task=ddf7fa4f"

# --- Code Golf Concepts ---
CONCEPTS = [
    "color_palette",
    "recoloring",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 2, 0, 0, 6, 0, 0, 0, 8],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 5, 5, 5, 5, 0, 0],
    [0, 0, 0, 0, 5, 5, 5, 5, 0, 0],
    [0, 5, 5, 0, 5, 5, 5, 5, 0, 0],
    [0, 5, 5, 0, 5, 5, 5, 5, 0, 0],
    [0, 5, 5, 0, 0, 0, 0, 0, 0, 0],
    [0, 5, 5, 0, 0, 0, 0, 5, 5, 5],
    [0, 5, 5, 0, 0, 0, 0, 5, 5, 5],
    [0, 0, 0, 0, 0, 0, 0, 5, 5, 5],
], dtype=int)

E1_OUT = np.array([
    [0, 0, 2, 0, 0, 6, 0, 0, 0, 8],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 6, 6, 6, 6, 0, 0],
    [0, 0, 0, 0, 6, 6, 6, 6, 0, 0],
    [0, 2, 2, 0, 6, 6, 6, 6, 0, 0],
    [0, 2, 2, 0, 6, 6, 6, 6, 0, 0],
    [0, 2, 2, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 2, 0, 0, 0, 0, 8, 8, 8],
    [0, 2, 2, 0, 0, 0, 0, 8, 8, 8],
    [0, 0, 0, 0, 0, 0, 0, 8, 8, 8],
], dtype=int)

E2_IN = np.array([
    [0, 1, 0, 0, 0, 4, 0, 0, 7, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [5, 5, 5, 5, 0, 0, 0, 5, 5, 5],
    [5, 5, 5, 5, 0, 0, 0, 5, 5, 5],
    [5, 5, 5, 5, 0, 0, 0, 5, 5, 5],
    [5, 5, 5, 5, 0, 0, 0, 5, 5, 5],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 5, 5, 5, 5, 0, 0, 0],
    [0, 0, 0, 5, 5, 5, 5, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [0, 1, 0, 0, 0, 4, 0, 0, 7, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 7, 7, 7],
    [1, 1, 1, 1, 0, 0, 0, 7, 7, 7],
    [1, 1, 1, 1, 0, 0, 0, 7, 7, 7],
    [1, 1, 1, 1, 0, 0, 0, 7, 7, 7],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 4, 4, 4, 4, 0, 0, 0],
    [0, 0, 0, 4, 4, 4, 4, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E3_IN = np.array([
    [0, 1, 0, 0, 0, 6, 0, 0, 7, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 5, 5, 0, 0, 0, 0, 0, 0, 0],
    [0, 5, 5, 0, 0, 0, 0, 5, 5, 5],
    [0, 5, 5, 0, 0, 0, 0, 5, 5, 5],
    [0, 0, 0, 5, 5, 5, 0, 0, 0, 0],
    [0, 0, 0, 5, 5, 5, 0, 0, 0, 0],
    [0, 0, 0, 5, 5, 5, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [0, 1, 0, 0, 0, 6, 0, 0, 7, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 7, 7, 7],
    [0, 1, 1, 0, 0, 0, 0, 7, 7, 7],
    [0, 0, 0, 6, 6, 6, 0, 0, 0, 0],
    [0, 0, 0, 6, 6, 6, 0, 0, 0, 0],
    [0, 0, 0, 6, 6, 6, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [3, 0, 0, 0, 6, 0, 0, 0, 9, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 5, 5, 5, 5, 5, 0, 5, 5],
    [0, 0, 5, 5, 5, 5, 5, 0, 5, 5],
    [0, 0, 5, 5, 5, 5, 5, 0, 5, 5],
    [0, 0, 5, 5, 5, 5, 5, 0, 5, 5],
    [0, 0, 0, 0, 0, 0, 0, 0, 5, 5],
    [5, 5, 5, 5, 0, 0, 0, 0, 5, 5],
    [5, 5, 5, 5, 0, 0, 0, 0, 5, 5],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

T_OUT = np.array([
    [3, 0, 0, 0, 6, 0, 0, 0, 9, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 6, 6, 6, 6, 6, 0, 9, 9],
    [0, 0, 6, 6, 6, 6, 6, 0, 9, 9],
    [0, 0, 6, 6, 6, 6, 6, 0, 9, 9],
    [0, 0, 6, 6, 6, 6, 6, 0, 9, 9],
    [0, 0, 0, 0, 0, 0, 0, 0, 9, 9],
    [3, 3, 3, 3, 0, 0, 0, 0, 9, 9],
    [3, 3, 3, 3, 0, 0, 0, 0, 9, 9],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
 A=range
 c=[x[:]for x in j]
 def d(E,k,W):
  if 0<=E<10 and 0<=k<10 and c[E][k]==5:c[E][k]=W;[d(E+a,k+b,W)for a,b in[(-1,0),(1,0),(0,-1),(0,1)]]
 [[d(E,k,j[0][k])for E in A(1,10)if c[E][k]==5]for k in A(10)if j[0][k]]
 return c


# --- Code Golf Solution (Compressed) ---
def q(g):
    return [[(c := [max(j and c, *g[0][j:], key=bool), x][x < 5]) for j, x in enumerate(r)] for r in g]


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, randint, uniform

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

ContainerContainer = Container[Container]

def identity(
    x: Any
) -> Any:
    """ identity function """
    return x

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

def generate_ddf7fa4f(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)  
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    nocc = unifint(diff_lb, diff_ub, (1, min(w // 3, (h * w) // 36)))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    succ = 0
    tr = 0
    maxtr = 10 * nocc
    inds = asindices(gi)
    inds = sfilter(inds, lambda ij: ij[0] > 1)
    while succ < nocc and tr < maxtr:
        tr += 1
        oh = randint(2, 7)
        ow = randint(2, 7)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        hastobein = {cidx for cidx, col in enumerate(gi[0]) if col == bgc}
        cantbein = {cidx for cidx, col in enumerate(gi[0]) if col != bgc}
        jopts = [j for j in range(w) if \
            len(set(interval(j, j + ow, 1)) & hastobein) > 0 and len(set(interval(j, j + ow, 1)) & cantbein) == 0
        ]
        cands = sfilter(cands, lambda ij: ij[1] in jopts)
        if len(cands) == 0:
            continue
        loci, locj = choice(totuple(cands))
        locat = choice(sfilter(interval(locj, locj + ow, 1), lambda jj: jj in hastobein))
        sq = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
        if sq.issubset(inds):
            succ += 1
            inds = (inds - sq) - mapply(dneighbors, sq)
            col = choice(remcols)
            gr = choice(remove(col, remcols))
            gi = fill(gi, col, {(0, locat)})
            go = fill(go, col, {(0, locat)})
            gi = fill(gi, gr, sq)
            go = fill(go, col, sq)
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Numerical = Union[Integer, IntegerTuple]

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

ZERO = 0

ONE = 1

TEN = 10

F = False

T = True

ORIGIN = (0, 0)

def flip(
    b: Boolean
) -> Boolean:
    """ logical not """
    return not b

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

def difference(
    a: Container,
    b: Container
) -> Container:
    """ difference """
    return type(a)(e for e in a if e not in b)

def size(
    container: Container
) -> Integer:
    """ cardinality """
    return len(container)

def argmax(
    container: Container,
    compfunc: Callable
) -> Any:
    """ largest item by custom order """
    return max(container, key=compfunc, default=None)

def initset(
    value: Any
) -> FrozenSet:
    """ initialize container """
    return frozenset({value})

def either(
    a: Boolean,
    b: Boolean
) -> Boolean:
    """ logical or """
    return a or b

def decrement(
    x: Numerical
) -> Numerical:
    """ decrementing """
    return x - 1 if isinstance(x, int) else (x[0] - 1, x[1] - 1)

def toivec(
    i: Integer
) -> IntegerTuple:
    """ vector pointing vertically """
    return (i, 0)

def tojvec(
    j: Integer
) -> IntegerTuple:
    """ vector pointing horizontally """
    return (0, j)

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

def sizefilter(
    container: Container,
    n: Integer
) -> FrozenSet:
    """ filter items by size """
    return frozenset(item for item in container if len(item) == n)

def ofcolor(
    grid: Grid,
    value: Integer
) -> Indices:
    """ indices of all grid cells with value """
    return frozenset((i, j) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value)

def crop(
    grid: Grid,
    start: IntegerTuple,
    dims: IntegerTuple
) -> Grid:
    """ subgrid specified by start and dimension """
    return tuple(r[start[1]:start[1]+dims[1]] for r in grid[start[0]:start[0]+dims[0]])

def recolor(
    value: Integer,
    patch: Patch
) -> Object:
    """ recolor patch """
    return frozenset((value, index) for index in toindices(patch))

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

def color(
    obj: Object
) -> Integer:
    """ color of object """
    return next(iter(obj))[0]

def toobject(
    patch: Patch,
    grid: Grid
) -> Object:
    """ object from patch and grid """
    h, w = len(grid), len(grid[0])
    return frozenset((grid[i][j], (i, j)) for i, j in toindices(patch) if 0 <= i < h and 0 <= j < w)

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

def subgrid(
    patch: Patch,
    grid: Grid
) -> Grid:
    """ smallest subgrid containing object """
    return crop(grid, ulcorner(patch), shape(patch))

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

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_ddf7fa4f(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = width(I)
    x1 = decrement(x0)
    x2 = tojvec(x1)
    x3 = connect(ORIGIN, x2)
    x4 = height(I)
    x5 = decrement(x4)
    x6 = toivec(x5)
    x7 = connect(ORIGIN, x6)
    x8 = width(I)
    x9 = decrement(x8)
    x10 = tojvec(x9)
    x11 = shape(I)
    x12 = decrement(x11)
    x13 = connect(x10, x12)
    x14 = height(I)
    x15 = decrement(x14)
    x16 = toivec(x15)
    x17 = shape(I)
    x18 = decrement(x17)
    x19 = connect(x16, x18)
    x20 = asindices(I)
    x21 = box(x20)
    x22 = toobject(x21, I)
    x23 = mostcolor(x22)
    x24 = matcher(color, x23)
    x25 = compose(flip, x24)
    x26 = rbind(sfilter, x25)
    x27 = rbind(sizefilter, ONE)
    x28 = rbind(objects, F)
    x29 = rbind(x28, F)
    x30 = rbind(x29, T)
    x31 = rbind(subgrid, I)
    x32 = chain(x26, x30, x31)
    x33 = chain(size, x27, x32)
    x34 = astuple(x3, x7)
    x35 = astuple(x13, x19)
    x36 = combine(x34, x35)
    x37 = argmax(x36, x33)
    x38 = rbind(toobject, I)
    x39 = compose(x38, initset)
    x40 = ofcolor(I, x23)
    x41 = difference(x37, x40)
    x42 = apply(x39, x41)
    x43 = rbind(intersection, x37)
    x44 = chain(size, x43, toindices)
    x45 = matcher(x44, ZERO)
    x46 = objects(I, T, F, T)
    x47 = sfilter(x46, x45)
    x48 = lbind(fork, either)
    x49 = lbind(lbind, hmatching)
    x50 = lbind(lbind, vmatching)
    x51 = fork(x48, x49, x50)
    x52 = lbind(chain, size)
    x53 = rbind(x52, x51)
    x54 = lbind(lbind, sfilter)
    x55 = compose(last, last)
    x56 = chain(x53, x54, x55)
    x57 = rbind(compose, x51)
    x58 = lbind(lbind, extract)
    x59 = compose(last, last)
    x60 = chain(x57, x58, x59)
    x61 = compose(first, last)
    x62 = rbind(matcher, ONE)
    x63 = compose(x62, x56)
    x64 = fork(sfilter, x61, x63)
    x65 = lbind(fork, recolor)
    x66 = lbind(x65, color)
    x67 = compose(x66, x60)
    x68 = fork(mapply, x67, x64)
    x69 = fork(combine, first, x68)
    x70 = compose(first, last)
    x71 = fork(difference, x70, x64)
    x72 = compose(last, last)
    x73 = fork(apply, x60, x64)
    x74 = fork(difference, x72, x73)
    x75 = fork(astuple, x71, x74)
    x76 = fork(astuple, x69, x75)
    x77 = difference(x42, x42)
    x78 = power(x76, TEN)
    x79 = astuple(x42, x47)
    x80 = astuple(x77, x79)
    x81 = x78(x80)
    x82 = first(x81)
    x83 = paint(I, x82)
    return x83


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_ddf7fa4f(inp)
        assert pred == _to_grid(expected), f"{name} failed"

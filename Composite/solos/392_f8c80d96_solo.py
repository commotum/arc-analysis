# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "f8c80d96"
SERIAL = "392"
URL    = "https://arcprize.org/play?task=f8c80d96"

# --- Code Golf Concepts ---
CONCEPTS = [
    "pattern_expansion",
    "background_filling",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [8, 8, 8, 8, 0, 8, 0, 0, 0, 0],
    [0, 0, 0, 8, 0, 8, 0, 0, 0, 0],
    [8, 8, 0, 8, 0, 8, 0, 0, 0, 0],
    [0, 8, 0, 8, 0, 8, 0, 0, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 5, 8],
    [5, 5, 5, 5, 5, 5, 5, 8, 5, 8],
    [8, 8, 8, 8, 8, 8, 5, 8, 5, 8],
    [5, 5, 5, 5, 5, 8, 5, 8, 5, 8],
    [8, 8, 8, 8, 5, 8, 5, 8, 5, 8],
    [5, 5, 5, 8, 5, 8, 5, 8, 5, 8],
    [8, 8, 5, 8, 5, 8, 5, 8, 5, 8],
    [5, 8, 5, 8, 5, 8, 5, 8, 5, 8],
], dtype=int)

E2_IN = np.array([
    [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [5, 1, 5, 5, 1, 5, 5, 1, 5, 5],
    [5, 1, 5, 5, 1, 5, 5, 1, 5, 5],
    [5, 1, 5, 5, 1, 5, 5, 1, 1, 1],
    [5, 1, 5, 5, 1, 5, 5, 5, 5, 5],
    [5, 1, 5, 5, 1, 5, 5, 5, 5, 5],
    [5, 1, 5, 5, 1, 1, 1, 1, 1, 1],
    [5, 1, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 1, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
], dtype=int)

E3_IN = np.array([
    [0, 2, 0, 2, 0, 2, 0, 2, 0, 0],
    [0, 2, 0, 2, 2, 2, 0, 2, 0, 0],
    [0, 2, 0, 0, 0, 0, 0, 2, 0, 0],
    [0, 2, 2, 2, 2, 2, 2, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [5, 2, 5, 2, 5, 2, 5, 2, 5, 2],
    [5, 2, 5, 2, 2, 2, 5, 2, 5, 2],
    [5, 2, 5, 5, 5, 5, 5, 2, 5, 2],
    [5, 2, 2, 2, 2, 2, 2, 2, 5, 2],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [4, 4, 4, 4, 4, 4, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
    [4, 4, 4, 0, 0, 4, 0, 0, 0, 0],
    [0, 0, 4, 0, 0, 4, 0, 0, 0, 0],
    [0, 0, 4, 0, 0, 4, 0, 0, 0, 0],
    [4, 4, 4, 0, 0, 4, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
    [4, 4, 4, 4, 4, 4, 0, 0, 0, 0],
], dtype=int)

T_OUT = np.array([
    [4, 4, 4, 4, 4, 4, 5, 5, 4, 5],
    [5, 5, 5, 5, 5, 4, 5, 5, 4, 5],
    [5, 5, 5, 5, 5, 4, 5, 5, 4, 5],
    [4, 4, 4, 5, 5, 4, 5, 5, 4, 5],
    [5, 5, 4, 5, 5, 4, 5, 5, 4, 5],
    [5, 5, 4, 5, 5, 4, 5, 5, 4, 5],
    [4, 4, 4, 5, 5, 4, 5, 5, 4, 5],
    [5, 5, 5, 5, 5, 4, 5, 5, 4, 5],
    [5, 5, 5, 5, 5, 4, 5, 5, 4, 5],
    [4, 4, 4, 4, 4, 4, 5, 5, 4, 5],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
L=len
R=range
def p(g):
 C=max(sum(g,[]))
 g=[[5 if c==0 else c for c in r] for r in g]
 return g


# --- Code Golf Solution (Compressed) ---
def q(g, n=23):
    return -n * g or [eval(f"r.pop()or[5,w:=max(max(g))][{n}//4%(('0, %d, '%w*2in'{g}')-3)]*any(r[-1:])," * 10) for *r, in zip(*p(g, n - 1))]


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import randint, sample, uniform

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Numerical = Union[Integer, IntegerTuple]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

def increment(
    x: Numerical
) -> Numerical:
    """ incrementing """
    return x + 1 if isinstance(x, int) else (x[0] + 1, x[1] + 1)

def decrement(
    x: Numerical
) -> Numerical:
    """ decrementing """
    return x - 1 if isinstance(x, int) else (x[0] - 1, x[1] - 1)

def sfilter(
    container: Container,
    condition: Callable
) -> Container:
    """ keep elements in container that satisfy condition """
    return type(container)(e for e in container if condition(e))

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

def generate_f8c80d96(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(5, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (8, 30))
    w = unifint(diff_lb, diff_ub, (8, 30))
    ow = randint(1, 3 if h > 10 else 2)
    oh = randint(1, 3 if w > 10 else 2)
    loci = randint(-oh+1, h-1)
    locj = randint(-ow+1, w-1)
    obj = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
    bgc, linc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    go = canvas(5, (h, w))
    ln1 = outbox(obj)
    ulci, ulcj = decrement(ulcorner(obj))
    lrci, lrcj = increment(lrcorner(obj))
    hoffs = randint(2, 4 if h > 12 else 3)
    woffs = randint(2, 4 if w > 12 else 3)
    lns = []
    for k in range(max(h, w) // min(hoffs, woffs) + 1):
        lnx = box(frozenset({(ulci - hoffs * k, ulcj - woffs * k), (lrci + hoffs * k, lrcj + woffs * k)}))
        lns.append(lnx)
    inds = asindices(gi)
    lns = sfilter(lns, lambda ln: len(ln & inds) > 0)
    nlns = len(lns)
    nmissing = unifint(diff_lb, diff_ub, (0, nlns - 2))
    npresent = nlns - nmissing
    for k in range(npresent):
        gi = fill(gi, linc, lns[k])
    for ln in lns:
        go = fill(go, linc, ln)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

IntegerSet = FrozenSet[Integer]

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

ContainerContainer = Container[Container]

ONE = 1

FIVE = 5

EIGHT = 8

F = False

T = True

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

def subtract(
    a: Numerical,
    b: Numerical
) -> Numerical:
    """ subtraction """
    if isinstance(a, int) and isinstance(b, int):
        return a - b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] - b[0], a[1] - b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a - b[0], a - b[1])
    return (a[0] - b, a[1] - b)

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

def double(
    n: Numerical
) -> Numerical:
    """ scaling by two """
    return n * 2 if isinstance(n, int) else (n[0] * 2, n[1] * 2)

def merge(
    containers: ContainerContainer
) -> Container:
    """ merging """
    return type(containers)(e for c in containers for e in c)

def maximum(
    container: IntegerSet
) -> Integer:
    """ maximum """
    return max(container, default=0)

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

def first(
    container: Container
) -> Any:
    """ first item of container """
    return next(iter(container))

def insert(
    value: Any,
    container: FrozenSet
) -> FrozenSet:
    """ insert item into container """
    return container.union(frozenset({value}))

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

def colorfilter(
    objs: Objects,
    value: Integer
) -> Objects:
    """ filter objects by color """
    return frozenset(obj for obj in objs if next(iter(obj))[0] == value)

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

def replace(
    grid: Grid,
    replacee: Integer,
    replacer: Integer
) -> Grid:
    """ color substitution """
    return tuple(tuple(replacer if v == replacee else v for v in r) for r in grid)

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_f8c80d96(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = objects(I, T, F, F)
    x1 = compose(maximum, shape)
    x2 = argmin(x0, x1)
    x3 = color(x2)
    x4 = palette(I)
    x5 = other(x4, x3)
    x6 = colorfilter(x0, x5)
    x7 = argmin(x6, x1)
    x8 = remove(x7, x6)
    x9 = rbind(manhattan, x7)
    x10 = argmin(x8, x9)
    x11 = rightmost(x10)
    x12 = rightmost(x7)
    x13 = subtract(x11, x12)
    x14 = leftmost(x7)
    x15 = leftmost(x10)
    x16 = subtract(x14, x15)
    x17 = astuple(x13, x16)
    x18 = maximum(x17)
    x19 = lowermost(x10)
    x20 = lowermost(x7)
    x21 = subtract(x19, x20)
    x22 = uppermost(x7)
    x23 = uppermost(x10)
    x24 = subtract(x22, x23)
    x25 = astuple(x21, x24)
    x26 = maximum(x25)
    x27 = ulcorner(x7)
    x28 = lrcorner(x7)
    x29 = astuple(x26, x18)
    x30 = double(EIGHT)
    x31 = interval(ONE, x30, ONE)
    x32 = lbind(subtract, x27)
    x33 = rbind(multiply, x29)
    x34 = compose(x32, x33)
    x35 = lbind(add, x28)
    x36 = rbind(multiply, x29)
    x37 = chain(initset, x35, x36)
    x38 = fork(insert, x34, x37)
    x39 = compose(box, x38)
    x40 = mapply(x39, x31)
    x41 = fill(I, x5, x40)
    x42 = replace(x41, x3, FIVE)
    return x42


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_f8c80d96(inp)
        assert pred == _to_grid(expected), f"{name} failed"

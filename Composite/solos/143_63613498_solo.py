# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "63613498"
SERIAL = "143"
URL    = "https://arcprize.org/play?task=63613498"

# --- Code Golf Concepts ---
CONCEPTS = [
    "recoloring",
    "compare_image",
    "detect_wall",
]

# --- Example Grids ---
E1_IN = np.array([
    [1, 1, 0, 5, 0, 0, 0, 7, 0, 0],
    [0, 1, 1, 5, 0, 0, 7, 7, 7, 0],
    [0, 0, 1, 5, 0, 0, 0, 7, 0, 0],
    [5, 5, 5, 5, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 6, 6, 0, 0, 0],
    [7, 7, 7, 0, 0, 0, 6, 6, 0, 0],
    [0, 0, 7, 0, 0, 0, 0, 6, 0, 0],
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 8, 8, 8, 0, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [1, 1, 0, 5, 0, 0, 0, 7, 0, 0],
    [0, 1, 1, 5, 0, 0, 7, 7, 7, 0],
    [0, 0, 1, 5, 0, 0, 0, 7, 0, 0],
    [5, 5, 5, 5, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 5, 5, 0, 0, 0],
    [7, 7, 7, 0, 0, 0, 5, 5, 0, 0],
    [0, 0, 7, 0, 0, 0, 0, 5, 0, 0],
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 8, 8, 8, 0, 0, 0],
], dtype=int)

E2_IN = np.array([
    [0, 3, 0, 5, 0, 0, 0, 0, 0, 0],
    [3, 3, 0, 5, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 4, 4, 0, 0],
    [5, 5, 5, 5, 0, 0, 4, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 7, 7, 7, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 9, 0],
    [8, 8, 0, 0, 0, 0, 0, 9, 9, 0],
    [8, 8, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [0, 3, 0, 5, 0, 0, 0, 0, 0, 0],
    [3, 3, 0, 5, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 4, 4, 0, 0],
    [5, 5, 5, 5, 0, 0, 4, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 7, 7, 7, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 5, 0],
    [8, 8, 0, 0, 0, 0, 0, 5, 5, 0],
    [8, 8, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E3_IN = np.array([
    [0, 4, 0, 5, 0, 0, 0, 1, 0, 0],
    [4, 4, 4, 5, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
    [5, 5, 5, 5, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 3, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 3, 3, 0, 0],
    [0, 0, 0, 0, 0, 0, 3, 0, 0, 0],
    [0, 6, 6, 6, 0, 0, 0, 0, 0, 0],
    [0, 0, 6, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [0, 4, 0, 5, 0, 0, 0, 5, 0, 0],
    [4, 4, 4, 5, 0, 0, 5, 5, 5, 0],
    [0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
    [5, 5, 5, 5, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 3, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 3, 3, 0, 0],
    [0, 0, 0, 0, 0, 0, 3, 0, 0, 0],
    [0, 6, 6, 6, 0, 0, 0, 0, 0, 0],
    [0, 0, 6, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 0, 5, 0, 0, 0, 6, 0, 0],
    [2, 2, 2, 5, 0, 0, 0, 6, 6, 0],
    [0, 0, 2, 5, 0, 0, 0, 6, 0, 0],
    [5, 5, 5, 5, 0, 3, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 3, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 3, 3, 0, 0, 0],
    [8, 8, 8, 0, 0, 0, 0, 0, 4, 0],
    [8, 0, 0, 0, 0, 0, 0, 0, 4, 0],
    [0, 0, 0, 0, 7, 7, 7, 0, 4, 0],
    [0, 0, 0, 0, 0, 0, 7, 0, 0, 0],
], dtype=int)

T_OUT = np.array([
    [0, 0, 0, 5, 0, 0, 0, 6, 0, 0],
    [2, 2, 2, 5, 0, 0, 0, 6, 6, 0],
    [0, 0, 2, 5, 0, 0, 0, 6, 0, 0],
    [5, 5, 5, 5, 0, 3, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 3, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 3, 3, 0, 0, 0],
    [8, 8, 8, 0, 0, 0, 0, 0, 4, 0],
    [8, 0, 0, 0, 0, 0, 0, 0, 4, 0],
    [0, 0, 0, 0, 5, 5, 5, 0, 4, 0],
    [0, 0, 0, 0, 0, 0, 5, 0, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(*args, **kwargs):
    raise NotImplementedError("Barnacles solution not available for 143")


# --- Code Golf Solution (Compressed) ---
def q(g, i=1):
    *R, = G = b'%r' % g
    for k in b'\x02\x05\x08"%(BEH':
        R[k + i] = (48 % G[k] or -5) % G[k + i] + 5
    return {*G} > {*R} and eval(bytes(R)) or p(g, i + 1)


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, randint, sample, uniform

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

Piece = Union[Grid, Patch]

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

def generate_63613498(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc, sepc = sample(cols, 2)
    remcols = remove(bgc, remove(sepc, cols))
    ncols = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, ncols)
    objh = unifint(diff_lb, diff_ub, (1, h//3))
    objw = unifint(diff_lb, diff_ub, (1, w//3))
    bounds = asindices(canvas(-1, (objh, objw)))
    sp = choice(totuple(bounds))
    obj = {sp}
    ncells = unifint(diff_lb, diff_ub, (1, (objh * objw)))
    for k in range(ncells - 1):
        obj.add(choice(totuple((bounds - obj) & mapply(dneighbors, obj))))
    gi = canvas(bgc, (h, w))
    objc = choice(ccols)
    gi = fill(gi, objc, obj)
    sep = connect((objh+1, 0), (objh+1, objw+1)) | connect((0, objw+1), (objh+1, objw+1))
    gi = fill(gi, sepc, sep)
    inds = asindices(gi)
    inds -= backdrop(sep)
    nobjs = unifint(diff_lb, diff_ub, (1, (h * w) // 20))
    succ = 0
    tr = 0
    maxtr = 5 * nobjs
    baseobj = normalize(obj)
    obj = normalize(obj)
    go = tuple(e for e in gi)
    while (succ < nobjs and tr < maxtr) or succ == 0:
        tr += 1
        oh, ow = shape(obj)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            break
        loc = choice(totuple(cands))
        plcd = shift(obj, loc)
        if plcd.issubset(inds):
            col = choice(ccols)
            gi = fill(gi, col, plcd)
            go = fill(go, sepc if succ == 0 else col, plcd)
            succ += 1
            inds = (inds - plcd) - mapply(dneighbors, plcd)
        objh = randint(1, h // 3)
        objw = randint(2 if objh == 1 else 1, w // 3)
        if choice((True, False)):
            objh, objw = objw, objh
        bounds = asindices(canvas(-1, (objh, objw)))
        sp = choice(totuple(bounds))
        obj = {sp}
        ncells = unifint(diff_lb, diff_ub, (1, (objh * objw)))
        for k in range(ncells - 1):
            obj.add(choice(totuple((bounds - obj) & mapply(dneighbors, obj))))
        obj = normalize(obj)
        obj = set(obj)
        if obj == baseobj:
            if len(obj) < objh * objw:
                obj.add(choice(totuple((bounds - obj) & mapply(dneighbors, obj))))
            else:
                obj = remove(choice(totuple(corners(obj))), obj)
        obj = normalize(obj)
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Numerical = Union[Integer, IntegerTuple]

IntegerSet = FrozenSet[Integer]

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

TWO = 2

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

def flip(
    b: Boolean
) -> Boolean:
    """ logical not """
    return not b

def equality(
    a: Any,
    b: Any
) -> Boolean:
    """ equality """
    return a == b

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

def both(
    a: Boolean,
    b: Boolean
) -> Boolean:
    """ logical and """
    return a and b

def decrement(
    x: Numerical
) -> Numerical:
    """ decrementing """
    return x - 1 if isinstance(x, int) else (x[0] - 1, x[1] - 1)

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

def bordering(
    patch: Patch,
    grid: Grid
) -> Boolean:
    """ whether a patch is adjacent to a grid border """
    return uppermost(patch) == 0 or leftmost(patch) == 0 or lowermost(patch) == len(grid) - 1 or rightmost(patch) == len(grid[0]) - 1

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

def toobject(
    patch: Patch,
    grid: Grid
) -> Object:
    """ object from patch and grid """
    h, w = len(grid), len(grid[0])
    return frozenset((grid[i][j], (i, j)) for i, j in toindices(patch) if 0 <= i < h and 0 <= j < w)

def delta(
    patch: Patch
) -> Indices:
    """ indices in bounding box but not part of patch """
    if len(patch) == 0:
        return frozenset({})
    return backdrop(patch) - toindices(patch)

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_63613498(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = objects(I, T, F, T)
    x1 = mostcolor(I)
    x2 = fork(add, height, width)
    x3 = compose(decrement, x2)
    x4 = fork(equality, x3, size)
    x5 = rbind(bordering, I)
    x6 = fork(both, x4, x5)
    x7 = rbind(toobject, I)
    x8 = chain(numcolors, x7, delta)
    x9 = matcher(x8, TWO)
    x10 = fork(both, x6, x9)
    x11 = sfilter(x0, x10)
    x12 = argmax(x11, size)
    x13 = delta(x12)
    x14 = toobject(x13, I)
    x15 = matcher(first, x1)
    x16 = compose(flip, x15)
    x17 = sfilter(x14, x16)
    x18 = normalize(x17)
    x19 = toindices(x18)
    x20 = compose(toindices, normalize)
    x21 = matcher(x20, x19)
    x22 = remove(x17, x0)
    x23 = argmax(x22, x21)
    x24 = color(x12)
    x25 = fill(I, x24, x23)
    return x25


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_63613498(inp)
        assert pred == _to_grid(expected), f"{name} failed"

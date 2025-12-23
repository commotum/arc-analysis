# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "228f6490"
SERIAL = "044"
URL    = "https://arcprize.org/play?task=228f6490"

# --- Code Golf Concepts ---
CONCEPTS = [
    "pattern_moving",
    "loop_filling",
    "shape_guessing",
    "x_marks_the_spot",
]

# --- Example Grids ---
E1_IN = np.array([
    [7, 0, 0, 0, 0, 0, 0, 0, 7, 7],
    [0, 5, 5, 5, 5, 5, 0, 0, 0, 0],
    [0, 5, 0, 0, 5, 5, 0, 6, 6, 0],
    [0, 5, 0, 0, 5, 5, 0, 0, 0, 0],
    [0, 5, 5, 5, 5, 5, 0, 0, 0, 0],
    [0, 5, 5, 5, 5, 5, 0, 0, 7, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 7, 5, 5, 5, 5, 5],
    [0, 8, 8, 0, 0, 5, 5, 0, 0, 5],
    [0, 8, 8, 0, 0, 5, 5, 5, 5, 5],
], dtype=int)

E1_OUT = np.array([
    [7, 0, 0, 0, 0, 0, 0, 0, 7, 7],
    [0, 5, 5, 5, 5, 5, 0, 0, 0, 0],
    [0, 5, 8, 8, 5, 5, 0, 0, 0, 0],
    [0, 5, 8, 8, 5, 5, 0, 0, 0, 0],
    [0, 5, 5, 5, 5, 5, 0, 0, 0, 0],
    [0, 5, 5, 5, 5, 5, 0, 0, 7, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 7, 5, 5, 5, 5, 5],
    [0, 0, 0, 0, 0, 5, 5, 6, 6, 5],
    [0, 0, 0, 0, 0, 5, 5, 5, 5, 5],
], dtype=int)

E2_IN = np.array([
    [5, 5, 5, 5, 5, 0, 0, 0, 0, 0],
    [5, 0, 0, 0, 5, 0, 9, 9, 9, 9],
    [5, 5, 5, 0, 5, 0, 9, 9, 9, 9],
    [5, 5, 5, 5, 5, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 6, 0, 6],
    [3, 3, 3, 0, 0, 0, 6, 6, 0, 0],
    [0, 0, 3, 5, 5, 5, 5, 5, 5, 0],
    [0, 0, 0, 5, 0, 0, 0, 0, 5, 0],
    [6, 6, 0, 5, 0, 0, 0, 0, 5, 0],
    [6, 6, 0, 5, 5, 5, 5, 5, 5, 0],
], dtype=int)

E2_OUT = np.array([
    [5, 5, 5, 5, 5, 0, 0, 0, 0, 0],
    [5, 3, 3, 3, 5, 0, 0, 0, 0, 0],
    [5, 5, 5, 3, 5, 0, 0, 0, 0, 0],
    [5, 5, 5, 5, 5, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 6, 0, 6],
    [0, 0, 0, 0, 0, 0, 6, 6, 0, 0],
    [0, 0, 0, 5, 5, 5, 5, 5, 5, 0],
    [0, 0, 0, 5, 9, 9, 9, 9, 5, 0],
    [6, 6, 0, 5, 9, 9, 9, 9, 5, 0],
    [6, 6, 0, 5, 5, 5, 5, 5, 5, 0],
], dtype=int)

E3_IN = np.array([
    [2, 2, 0, 0, 5, 5, 5, 5, 5, 5],
    [2, 2, 2, 0, 5, 0, 0, 0, 5, 5],
    [0, 0, 0, 0, 5, 5, 5, 0, 0, 5],
    [0, 4, 4, 0, 5, 5, 5, 5, 5, 5],
    [0, 0, 4, 0, 0, 4, 0, 0, 0, 0],
    [5, 5, 5, 5, 5, 0, 0, 4, 4, 0],
    [5, 5, 5, 5, 5, 0, 0, 0, 0, 0],
    [5, 0, 0, 5, 5, 0, 0, 0, 0, 4],
    [5, 0, 0, 0, 5, 0, 8, 8, 8, 0],
    [5, 5, 5, 5, 5, 0, 0, 0, 8, 8],
], dtype=int)

E3_OUT = np.array([
    [0, 0, 0, 0, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 0, 5, 8, 8, 8, 5, 5],
    [0, 0, 0, 0, 5, 5, 5, 8, 8, 5],
    [0, 4, 4, 0, 5, 5, 5, 5, 5, 5],
    [0, 0, 4, 0, 0, 4, 0, 0, 0, 0],
    [5, 5, 5, 5, 5, 0, 0, 4, 4, 0],
    [5, 5, 5, 5, 5, 0, 0, 0, 0, 0],
    [5, 2, 2, 5, 5, 0, 0, 0, 0, 4],
    [5, 2, 2, 2, 5, 0, 0, 0, 0, 0],
    [5, 5, 5, 5, 5, 0, 0, 0, 0, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 5, 5, 5, 5, 5, 0, 0, 2],
    [2, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 0, 5, 5, 0, 5, 5, 4, 4, 4],
    [0, 0, 5, 5, 5, 5, 5, 0, 0, 0],
    [0, 0, 5, 5, 5, 5, 5, 0, 0, 2],
    [7, 7, 7, 0, 0, 2, 0, 2, 0, 0],
    [0, 7, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 5, 5, 5, 5, 5, 5, 5],
    [0, 2, 0, 5, 0, 0, 0, 5, 5, 5],
    [2, 0, 0, 5, 5, 5, 5, 5, 5, 5],
], dtype=int)

T_OUT = np.array([
    [0, 0, 5, 5, 5, 5, 5, 0, 0, 2],
    [2, 0, 5, 7, 7, 7, 5, 0, 0, 0],
    [0, 0, 5, 5, 7, 5, 5, 0, 0, 0],
    [0, 0, 5, 5, 5, 5, 5, 0, 0, 0],
    [0, 0, 5, 5, 5, 5, 5, 0, 0, 2],
    [0, 0, 0, 0, 0, 2, 0, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 5, 5, 5, 5, 5, 5, 5],
    [0, 2, 0, 5, 4, 4, 4, 5, 5, 5],
    [2, 0, 0, 5, 5, 5, 5, 5, 5, 5],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(*args, **kwargs):
    raise NotImplementedError("Barnacles solution not available for 044")


# --- Code Golf Solution (Compressed) ---
def q(i):
    for f in range(100):
        r = [n for n in range(100) if i[n // 10][n % 10] < 5 in i[n // 10][:n % 10] and 5 in i[n // 10][n % 10:] and (n < 50)]
        t = [n for n in range(100) if i[n // 10][n % 10] == f]
        if [r[0] - f for f in r] == [t[0] - f for f in t]:
            for n in r:
                i[n // 10][n % 10] = f
            for n in t:
                i[n // 10][n % 10] = 0
    for f in range(100):
        r = [n for n in range(100) if i[n // 10][n % 10] < 5 in i[n // 10][:n % 10] and 5 in i[n // 10][n % 10:] and (n > 50)]
        t = [n for n in range(100) if i[n // 10][n % 10] == f]
        if [r[0] - f for f in r] == [t[0] - f for f in t]:
            for n in r:
                i[n // 10][n % 10] = f
            for n in t:
                i[n // 10][n % 10] = 0
    return i


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

Piece = Union[Grid, Patch]

ContainerContainer = Container[Container]

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

def generate_228f6490(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    nsq = unifint(diff_lb, diff_ub, (1, (h * w) // 50))
    succ = 0
    tr = 0
    maxtr = 5 * nsq
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    sqc = choice(remcols)
    remcols = remove(sqc, remcols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    forbidden = []
    while tr < maxtr and succ < nsq:
        tr += 1
        oh = randint(3, 6)
        ow = randint(3, 6)
        bd = asindices(canvas(-1, (oh, ow)))
        bounds = shift(asindices(canvas(-1, (oh-2, ow-2))), (1, 1))
        obj = {choice(totuple(bounds))}
        ncells = randint(1, (oh-2) * (ow-2))
        for k in range(ncells - 1):
            obj.add(choice(totuple((bounds - obj) & mapply(dneighbors, obj))))
        sqcands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(sqcands) == 0:
            continue
        loc = choice(totuple(sqcands))
        bdplcd = shift(bd, loc)
        if bdplcd.issubset(inds):
            tmpinds = inds - bdplcd
            inobjn = normalize(obj)
            oh, ow = shape(obj)
            inobjcands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
            if len(inobjcands) == 0:
                continue
            loc2 = choice(totuple(inobjcands))
            inobjplcd = shift(inobjn, loc2)
            bdnorm = bd - obj
            if inobjplcd.issubset(tmpinds) and bdnorm not in forbidden and inobjn not in forbidden:
                forbidden.append(bdnorm)
                forbidden.append(inobjn)
                succ += 1
                inds = (inds - (bdplcd | inobjplcd)) - mapply(dneighbors, inobjplcd)
                col = choice(remcols)
                oplcd = shift(obj, loc)
                gi = fill(gi, sqc, bdplcd - oplcd)
                go = fill(go, sqc, bdplcd)
                go = fill(go, col, oplcd)
                gi = fill(gi, col, inobjplcd)
    nremobjs = unifint(diff_lb, diff_ub, (0, len(inds) // 25))
    succ = 0
    tr = 0
    maxtr = 10 * nremobjs
    while tr < maxtr and succ < nremobjs:
        tr += 1
        oh = randint(1, 4)
        ow = randint(1, 4)
        bounds = asindices(canvas(-1, (oh, ow)))
        obj = {choice(totuple(bounds))}
        ncells = randint(1, oh * ow)
        for k in range(ncells - 1):
            obj.add(choice(totuple((bounds - obj) & mapply(dneighbors, obj))))
        obj = normalize(obj)
        if obj in forbidden:
            continue
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        plcd = shift(obj, loc)
        if plcd.issubset(inds):
            succ += 1
            inds = (inds - plcd) - mapply(dneighbors, plcd)
            col = choice(remcols)
            gi = fill(gi, col, plcd)
            go = fill(go, col, plcd)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

F = False

T = True

def identity(
    x: Any
) -> Any:
    """ identity function """
    return x

def flip(
    b: Boolean
) -> Boolean:
    """ logical not """
    return not b

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

def mostcommon(
    container: Container
) -> Any:
    """ most common item """
    return max(set(container), key=container.count)

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

def colorfilter(
    objs: Objects,
    value: Integer
) -> Objects:
    """ filter objects by color """
    return frozenset(obj for obj in objs if next(iter(obj))[0] == value)

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

def bordering(
    patch: Patch,
    grid: Grid
) -> Boolean:
    """ whether a patch is adjacent to a grid border """
    return uppermost(patch) == 0 or leftmost(patch) == 0 or lowermost(patch) == len(grid) - 1 or rightmost(patch) == len(grid[0]) - 1

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

def cover(
    grid: Grid,
    patch: Patch
) -> Grid:
    """ remove object from grid """
    return fill(grid, mostcolor(grid), toindices(patch))

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_228f6490(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = mostcolor(I)
    x1 = objects(I, T, T, F)
    x2 = colorfilter(x1, x0)
    x3 = compose(normalize, toindices)
    x4 = difference(x1, x2)
    x5 = rbind(bordering, I)
    x6 = compose(flip, x5)
    x7 = sfilter(x2, x6)
    x8 = rbind(toobject, I)
    x9 = lbind(mapply, neighbors)
    x10 = compose(x9, toindices)
    x11 = fork(difference, x10, identity)
    x12 = chain(mostcolor, x8, x11)
    x13 = totuple(x7)
    x14 = apply(x12, x13)
    x15 = mostcommon(x14)
    x16 = matcher(x12, x15)
    x17 = sfilter(x7, x16)
    x18 = lbind(argmax, x4)
    x19 = lbind(matcher, x3)
    x20 = chain(x18, x19, x3)
    x21 = compose(color, x20)
    x22 = fork(recolor, x21, identity)
    x23 = mapply(x20, x17)
    x24 = cover(I, x23)
    x25 = mapply(x22, x17)
    x26 = paint(x24, x25)
    return x26


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_228f6490(inp)
        assert pred == _to_grid(expected), f"{name} failed"

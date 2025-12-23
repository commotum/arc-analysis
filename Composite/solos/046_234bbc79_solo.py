# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "234bbc79"
SERIAL = "046"
URL    = "https://arcprize.org/play?task=234bbc79"

# --- Code Golf Concepts ---
CONCEPTS = [
    "recoloring",
    "bring_patterns_close",
    "crop",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 5, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 0, 5, 1, 0, 5, 2, 2],
    [0, 0, 0, 0, 5, 0, 0, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [0, 2, 1, 1, 0, 0, 0],
    [2, 2, 0, 1, 2, 2, 2],
    [0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E2_IN = np.array([
    [0, 0, 0, 5, 1, 5, 0, 0, 0, 0, 0],
    [2, 2, 0, 0, 0, 0, 0, 0, 3, 3, 3],
    [0, 5, 0, 0, 0, 0, 0, 5, 3, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 0, 0, 0, 0, 3, 3, 3],
    [0, 2, 1, 1, 1, 3, 3, 0, 0],
], dtype=int)

E3_IN = np.array([
    [0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0],
    [2, 2, 2, 0, 5, 8, 8, 0, 0, 0, 0],
    [0, 0, 5, 0, 0, 0, 0, 0, 5, 6, 6],
], dtype=int)

E3_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 2, 0, 0, 8, 6, 6, 6],
    [0, 0, 2, 8, 8, 8, 0, 0, 0],
], dtype=int)

E4_IN = np.array([
    [0, 1, 5, 0, 0, 0, 0, 0, 2, 2, 0],
    [1, 1, 0, 0, 5, 2, 0, 5, 2, 0, 0],
    [0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0],
], dtype=int)

E4_OUT = np.array([
    [0, 1, 1, 2, 2, 0, 2, 2],
    [1, 1, 0, 0, 2, 2, 2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 5, 0, 5, 1, 0, 0, 5, 0, 5, 8],
    [2, 2, 0, 0, 1, 0, 5, 3, 0, 0, 8],
    [0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
], dtype=int)

T_OUT = np.array([
    [0, 2, 1, 1, 0, 0, 0, 0],
    [2, 2, 0, 1, 0, 3, 8, 8],
    [0, 0, 0, 1, 3, 3, 0, 8],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(*args, **kwargs):
    raise NotImplementedError("Barnacles solution not available for 046")


# --- Code Golf Solution (Compressed) ---
def q(g, d=3):
    g = ((5,), *zip(*g))
    return (*zip(*[[sum({*x * (q + r + s)} - {5}) for x in 3 * r][d:3 + d] for q, r, s in zip(g, g[1:], g[2:] + g) if any(r) or (d := (d - [*q, 5].index(5) + s.index(5))) * 0]),)


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, randint, sample, uniform

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Numerical = Union[Integer, IntegerTuple]

IntegerSet = FrozenSet[Integer]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Objects = FrozenSet[Object]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

ContainerContainer = Container[Container]

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

def sfilter(
    container: Container,
    condition: Callable
) -> Container:
    """ keep elements in container that satisfy condition """
    return type(container)(e for e in container if condition(e))

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

def recolor(
    value: Integer,
    patch: Patch
) -> Object:
    """ recolor patch """
    return frozenset((value, index) for index in toindices(patch))

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

def fgpartition(
    grid: Grid
) -> Objects:
    """ each cell with the same value part of the same object without background """
    return frozenset(
        frozenset(
            (v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value
        ) for value in palette(grid) - {mostcolor(grid)}
    )

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

def asobject(
    grid: Grid
) -> Object:
    """ conversion of grid to object """
    return frozenset((v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r))

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

def generate_234bbc79(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    while True:
        h = unifint(diff_lb, diff_ub, (5, 30))
        w = unifint(diff_lb, diff_ub, (6, 20))
        bgc, dotc = sample(cols, 2)
        remcols = difference(cols, (bgc, dotc))
        go = canvas(bgc, (h, 30))
        ncols = unifint(diff_lb, diff_ub, (1, 8))
        ccols = sample(remcols, ncols)
        spi = randint(0, h - 1)
        snek = [(spi, 0)]
        gi = fill(go, dotc, {(spi, 0)})
        while True:
            previ, prevj = snek[-1]
            if prevj == w - 1:
                if choice((True, False, False)):
                    break
            options = []
            if previ < h - 1:
                if go[previ+1][prevj] == bgc:
                    options.append((previ+1, prevj))
            if previ > 0:
                if go[previ-1][prevj] == bgc:
                    options.append((previ-1, prevj))
            if prevj < w - 1:
                options.append((previ, prevj+1))
            if len(options) == 0:
                break
            loc = choice(options)
            snek.append(loc)
            go = fill(go, dotc, {loc})
        objs = []
        cobj = []
        for idx, cel in enumerate(snek):
            if len(cobj) > 2 and width(frozenset(cobj)) > 1 and snek[idx-1] == add(cel, (0, -1)):
                objs.append(cobj)
                cobj = [cel]
            else:
                cobj.append(cel)
        objs[-1] += cobj
        nobjs = len(objs)
        if nobjs < 2:
            continue
        ntokeep = unifint(diff_lb, diff_ub, (2, nobjs))
        ntorem = nobjs - ntokeep
        for k in range(ntorem):
            idx = randint(0, len(objs) - 2)
            objs = objs[:idx] + [objs[idx] + objs[idx+1]] + objs[idx+2:]
        inobjs = []
        for idx, obj in enumerate(objs):
            col = choice(ccols)
            go = fill(go, col, set(obj))
            centerpart = recolor(col, set(obj[1:-1]))
            leftpart = {(dotc if idx > 0 else col, obj[0])}
            rightpart = {(dotc if idx < len(objs) - 1 else col, obj[-1])}
            inobj = centerpart | leftpart | rightpart
            inobjs.append(inobj)
        spacings = [1 for idx in range(len(inobjs) - 1)]
        fullw = unifint(diff_lb, diff_ub, (w, 30))
        for k in range(fullw - w - len(inobjs) - 1):
            idx = randint(0, len(spacings) - 1)
            spacings[idx] += 1
        lspacings = [0] + spacings
        gi = canvas(bgc, (h, fullw))
        ofs = 0
        for i, (lsp, obj) in enumerate(zip(lspacings, inobjs)):
            obj = set(obj)
            if i == 0:
                ulc = ulcorner(obj)
            else:
                ulci = randint(0, h - height(obj))
                ulcj = ofs + lsp
                ulc = (ulci, ulcj)
            ofs += width(obj) + lsp
            plcd = shift(normalize(obj), ulc)
            gi = paint(gi, plcd)
        break
    ins = size(merge(fgpartition(gi)))
    while True:
        go2 = dmirror(dmirror(go)[:-1])
        if size(sfilter(asobject(go2), lambda cij: cij[0] != bgc)) < ins:
            break
        else:
            go = go2
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

ONE = 1

TWO = 2

F = False

T = True

ORIGIN = (0, 0)

RIGHT = (0, 1)

def identity(
    x: Any
) -> Any:
    """ identity function """
    return x

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

def combine(
    a: Container,
    b: Container
) -> Container:
    """ union """
    return type(a)((*a, *b))

def order(
    container: Container,
    compfunc: Callable
) -> Tuple:
    """ order container by custom key """
    return tuple(sorted(container, key=compfunc))

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

def decrement(
    x: Numerical
) -> Numerical:
    """ decrementing """
    return x - 1 if isinstance(x, int) else (x[0] - 1, x[1] - 1)

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

def branch(
    condition: Boolean,
    if_value: Any,
    else_value: Any
) -> Any:
    """ if else branching """
    return if_value if condition else else_value

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

def colorcount(
    element: Element,
    value: Integer
) -> Integer:
    """ number of cells with color """
    if isinstance(element, tuple):
        return sum(row.count(value) for row in element)
    return sum(v == value for v, _ in element)

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

def crop(
    grid: Grid,
    start: IntegerTuple,
    dims: IntegerTuple
) -> Grid:
    """ subgrid specified by start and dimension """
    return tuple(r[start[1]:start[1]+dims[1]] for r in grid[start[0]:start[0]+dims[0]])

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

def toobject(
    patch: Patch,
    grid: Grid
) -> Object:
    """ object from patch and grid """
    h, w = len(grid), len(grid[0])
    return frozenset((grid[i][j], (i, j)) for i, j in toindices(patch) if 0 <= i < h and 0 <= j < w)

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

def verify_234bbc79(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = objects(I, F, F, T)
    x1 = order(x0, leftmost)
    x2 = astuple(ONE, TWO)
    x3 = rbind(contained, x2)
    x4 = lbind(compose, x3)
    x5 = lbind(rbind, colorcount)
    x6 = compose(x4, x5)
    x7 = lbind(sfilter, x0)
    x8 = chain(size, x7, x6)
    x9 = size(x0)
    x10 = matcher(x8, x9)
    x11 = palette(I)
    x12 = sfilter(x11, x10)
    x13 = lbind(colorcount, I)
    x14 = argmin(x12, x13)
    x15 = matcher(first, x14)
    x16 = rbind(extract, x15)
    x17 = compose(x16, first)
    x18 = fork(remove, x17, first)
    x19 = rbind(compose, initset)
    x20 = lbind(rbind, manhattan)
    x21 = compose(initset, x17)
    x22 = chain(x19, x20, x21)
    x23 = fork(argmin, x18, x22)
    x24 = compose(last, x17)
    x25 = compose(first, x23)
    x26 = fork(astuple, x25, x24)
    x27 = fork(insert, x26, x18)
    x28 = compose(last, last)
    x29 = rbind(argmin, x28)
    x30 = rbind(sfilter, x15)
    x31 = compose(first, last)
    x32 = chain(x29, x30, x31)
    x33 = compose(flip, x15)
    x34 = rbind(sfilter, x33)
    x35 = compose(first, last)
    x36 = fork(remove, x32, x35)
    x37 = compose(x34, x36)
    x38 = rbind(compose, initset)
    x39 = lbind(rbind, manhattan)
    x40 = compose(initset, x32)
    x41 = chain(x38, x39, x40)
    x42 = fork(argmin, x37, x41)
    x43 = compose(first, x42)
    x44 = compose(last, x32)
    x45 = fork(astuple, x43, x44)
    x46 = compose(first, last)
    x47 = fork(remove, x32, x46)
    x48 = fork(insert, x45, x47)
    x49 = rbind(shift, RIGHT)
    x50 = compose(last, x32)
    x51 = fork(subtract, x24, x50)
    x52 = fork(shift, x48, x51)
    x53 = compose(x49, x52)
    x54 = fork(combine, x27, x53)
    x55 = compose(first, last)
    x56 = fork(remove, x55, last)
    x57 = fork(astuple, x54, x56)
    x58 = size(x0)
    x59 = decrement(x58)
    x60 = power(x57, x59)
    x61 = first(x1)
    x62 = remove(x61, x1)
    x63 = astuple(x61, x62)
    x64 = x60(x63)
    x65 = first(x64)
    x66 = merge(x0)
    x67 = cover(I, x66)
    x68 = paint(x67, x65)
    x69 = height(I)
    x70 = width(x65)
    x71 = astuple(x69, x70)
    x72 = crop(x68, ORIGIN, x71)
    x73 = ofcolor(x72, x14)
    x74 = mostcolor(I)
    x75 = palette(x72)
    x76 = contained(x14, x75)
    x77 = matcher(first, x74)
    x78 = compose(flip, x77)
    x79 = rbind(sfilter, x78)
    x80 = mapply(dneighbors, x73)
    x81 = lbind(toobject, x80)
    x82 = compose(x79, x81)
    x83 = rbind(recolor, x73)
    x84 = chain(x83, mostcolor, x82)
    x85 = fork(paint, identity, x84)
    x86 = branch(x76, x85, identity)
    x87 = x86(x72)
    return x87


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("E4", E4_IN, E4_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_234bbc79(inp)
        assert pred == _to_grid(expected), f"{name} failed"

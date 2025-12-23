# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "ef135b50"
SERIAL = "381"
URL    = "https://arcprize.org/play?task=ef135b50"

# --- Code Golf Concepts ---
CONCEPTS = [
    "draw_line_from_point",
    "bridges",
    "connect_the_dots",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 2, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 2, 0, 0, 0, 0, 2, 2, 0],
    [2, 2, 2, 0, 0, 0, 0, 2, 2, 0],
    [0, 0, 0, 0, 0, 0, 0, 2, 2, 0],
    [0, 0, 0, 2, 2, 0, 0, 2, 2, 0],
    [0, 0, 0, 2, 2, 0, 0, 2, 2, 0],
    [0, 0, 0, 2, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 2, 0, 0, 0, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 2, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 2, 9, 9, 9, 9, 2, 2, 0],
    [2, 2, 2, 9, 9, 9, 9, 2, 2, 0],
    [0, 0, 0, 0, 0, 0, 0, 2, 2, 0],
    [0, 0, 0, 2, 2, 9, 9, 2, 2, 0],
    [0, 0, 0, 2, 2, 9, 9, 2, 2, 0],
    [0, 0, 0, 2, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 2, 0, 0, 0, 0, 0],
], dtype=int)

E2_IN = np.array([
    [2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 0, 0, 0, 0, 0, 2, 2, 2],
    [2, 2, 0, 0, 0, 0, 0, 2, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 2, 2, 2],
    [0, 0, 0, 2, 2, 0, 0, 2, 2, 2],
    [0, 0, 0, 2, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 2, 0, 2, 2, 2, 2],
    [0, 0, 0, 0, 0, 0, 2, 2, 2, 2],
], dtype=int)

E2_OUT = np.array([
    [2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 9, 9, 9, 9, 9, 2, 2, 2],
    [2, 2, 9, 9, 9, 9, 9, 2, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 2, 2, 2],
    [0, 0, 0, 2, 2, 9, 9, 2, 2, 2],
    [0, 0, 0, 2, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 2, 9, 2, 2, 2, 2],
    [0, 0, 0, 0, 0, 0, 2, 2, 2, 2],
], dtype=int)

E3_IN = np.array([
    [0, 0, 0, 0, 0, 0, 2, 2, 2, 2],
    [2, 2, 2, 2, 0, 0, 2, 2, 2, 2],
    [2, 2, 2, 2, 0, 0, 0, 0, 0, 0],
    [2, 2, 2, 2, 0, 2, 2, 2, 0, 0],
    [0, 0, 0, 0, 0, 2, 2, 2, 0, 0],
    [0, 0, 0, 0, 0, 2, 2, 2, 0, 0],
    [0, 0, 0, 0, 0, 2, 2, 2, 0, 2],
    [2, 2, 2, 2, 0, 2, 2, 2, 0, 2],
    [2, 2, 2, 2, 0, 2, 2, 2, 0, 2],
    [2, 2, 2, 2, 0, 0, 0, 0, 0, 2],
], dtype=int)

E3_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 2, 2, 2, 2],
    [2, 2, 2, 2, 9, 9, 2, 2, 2, 2],
    [2, 2, 2, 2, 0, 0, 0, 0, 0, 0],
    [2, 2, 2, 2, 9, 2, 2, 2, 0, 0],
    [0, 0, 0, 0, 0, 2, 2, 2, 0, 0],
    [0, 0, 0, 0, 0, 2, 2, 2, 0, 0],
    [0, 0, 0, 0, 0, 2, 2, 2, 9, 2],
    [2, 2, 2, 2, 9, 2, 2, 2, 9, 2],
    [2, 2, 2, 2, 9, 2, 2, 2, 9, 2],
    [2, 2, 2, 2, 0, 0, 0, 0, 0, 2],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [2, 2, 2, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 2, 0, 0, 0, 2, 2, 2, 2],
    [0, 0, 0, 0, 0, 0, 2, 2, 2, 2],
    [0, 2, 2, 2, 0, 0, 2, 2, 2, 2],
    [0, 2, 2, 2, 0, 0, 0, 0, 0, 0],
    [0, 2, 2, 2, 0, 2, 2, 2, 2, 0],
    [0, 2, 2, 2, 0, 2, 2, 2, 2, 0],
    [0, 2, 2, 2, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

T_OUT = np.array([
    [2, 2, 2, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 2, 9, 9, 9, 2, 2, 2, 2],
    [0, 0, 0, 0, 0, 0, 2, 2, 2, 2],
    [0, 2, 2, 2, 9, 9, 2, 2, 2, 2],
    [0, 2, 2, 2, 0, 0, 0, 0, 0, 0],
    [0, 2, 2, 2, 9, 2, 2, 2, 2, 0],
    [0, 2, 2, 2, 9, 2, 2, 2, 2, 0],
    [0, 2, 2, 2, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j,A=range):
 c=len(j)
 for E in A(1,c-1):
  k=W=0
  for l in A(c):
   J=j[E][l];k=[k,1][k<1 and J>1]
   if k==1 and J<1:k=2;W=[W,l][~W]
   if k>1 and J>1:
    for a in A(W,l):j[E][a]=9;k=1;W=0
 return j


# --- Code Golf Solution (Compressed) ---
def q(g):
    return [r * (P := (r in g[::9])) or [(P := (r.pop(0) or any(r * P) * 9)) for _ in g] for r in g]


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, randint, sample, uniform

Boolean = bool

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

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

def generate_ef135b50(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(9, interval(0, 10, 1))
    while True:
        h = unifint(diff_lb, diff_ub, (8, 30))
        w = unifint(diff_lb, diff_ub, (8, 30))
        bgc = choice(cols)
        remcols = remove(bgc, cols)
        numc = unifint(diff_lb, diff_ub, (1, 8))
        ccols = sample(remcols, numc)
        gi = canvas(bgc, (h, w))
        nsq = unifint(diff_lb, diff_ub, (2, (h * w) // 30))
        succ = 0
        tr = 0
        maxtr = 5 * nsq
        inds = asindices(gi)
        pats = set()
        while tr < maxtr and succ < nsq:
            tr += 1
            oh = randint(1, (h//3*2))
            ow = randint(1, (w//3*2))
            cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
            if len(cands) == 0:
                continue
            loc = choice(totuple(cands))
            loci, locj = loc
            bd = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
            if bd.issubset(inds):
                succ += 1
                inds = (inds - bd) - mapply(neighbors, bd)
                gi = fill(gi, choice(ccols), bd)
                pats.add(bd)
        res = set()
        ofc = ofcolor(gi, bgc)
        for pat1 in pats:
            for pat2 in remove(pat1, pats):
                if hmatching(pat1, pat2):
                    um = max(uppermost(pat1), uppermost(pat2))
                    bm = min(lowermost(pat1), lowermost(pat2))
                    lm = min(rightmost(pat1), rightmost(pat2)) + 1
                    rm = max(leftmost(pat1), leftmost(pat2)) - 1
                    res = res | backdrop(frozenset({(um, lm), (bm, rm)}))
        res = (res & ofc) - box(asindices(gi))
        go = fill(gi, 9, res)
        if go != gi:
            break
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Numerical = Union[Integer, IntegerTuple]

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

NINE = 9

F = False

T = True

NEG_ONE = -1

def identity(
    x: Any
) -> Any:
    """ identity function """
    return x

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

def color(
    obj: Object
) -> Integer:
    """ color of object """
    return next(iter(obj))[0]

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

def vsplit(
    grid: Grid,
    n: Integer
) -> Tuple:
    """ split grid vertically """
    h, w = len(grid) // n, len(grid[0])
    offset = len(grid) % n != 0
    return tuple(crop(grid, (h * i + i * offset, 0), (h, w)) for i in range(n))

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_ef135b50(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = objects(I, T, F, F)
    x1 = fork(multiply, height, width)
    x2 = fork(equality, size, x1)
    x3 = compose(flip, x2)
    x4 = sfilter(x0, x3)
    x5 = argmax(x4, x1)
    x6 = color(x5)
    x7 = ofcolor(I, x6)
    x8 = asindices(I)
    x9 = difference(x8, x7)
    x10 = fill(I, NEG_ONE, x9)
    x11 = lbind(recolor, NEG_ONE)
    x12 = rbind(ofcolor, NEG_ONE)
    x13 = chain(x11, backdrop, x12)
    x14 = fork(paint, identity, x13)
    x15 = height(x10)
    x16 = vsplit(x10, x15)
    x17 = mapply(x14, x16)
    x18 = ofcolor(x17, NEG_ONE)
    x19 = asindices(I)
    x20 = box(x19)
    x21 = difference(x18, x20)
    x22 = intersection(x21, x7)
    x23 = fill(I, NINE, x22)
    return x23


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_ef135b50(inp)
        assert pred == _to_grid(expected), f"{name} failed"

# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "941d9a10"
SERIAL = "226"
URL    = "https://arcprize.org/play?task=941d9a10"

# --- Code Golf Concepts ---
CONCEPTS = [
    "detect_grid",
    "loop_filling",
    "pairwise_analogy",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 5, 0, 0, 0, 0, 5, 0, 0],
    [0, 0, 5, 0, 0, 0, 0, 5, 0, 0],
    [0, 0, 5, 0, 0, 0, 0, 5, 0, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 5, 0, 0, 0, 0, 5, 0, 0],
    [0, 0, 5, 0, 0, 0, 0, 5, 0, 0],
    [0, 0, 5, 0, 0, 0, 0, 5, 0, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 5, 0, 0, 0, 0, 5, 0, 0],
    [0, 0, 5, 0, 0, 0, 0, 5, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [1, 1, 5, 0, 0, 0, 0, 5, 0, 0],
    [1, 1, 5, 0, 0, 0, 0, 5, 0, 0],
    [1, 1, 5, 0, 0, 0, 0, 5, 0, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 5, 2, 2, 2, 2, 5, 0, 0],
    [0, 0, 5, 2, 2, 2, 2, 5, 0, 0],
    [0, 0, 5, 2, 2, 2, 2, 5, 0, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 5, 0, 0, 0, 0, 5, 3, 3],
    [0, 0, 5, 0, 0, 0, 0, 5, 3, 3],
], dtype=int)

E2_IN = np.array([
    [0, 0, 0, 5, 0, 0, 0, 0, 5, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 5, 0, 0, 0, 0, 5, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 5, 0, 0, 0, 0, 5, 0],
    [0, 0, 0, 5, 0, 0, 0, 0, 5, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 5, 0, 0, 0, 0, 5, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 5, 0, 0, 0, 0, 5, 0],
], dtype=int)

E2_OUT = np.array([
    [1, 1, 1, 5, 0, 0, 0, 0, 5, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 5, 0, 0, 0, 0, 5, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 5, 2, 2, 2, 2, 5, 0],
    [0, 0, 0, 5, 2, 2, 2, 2, 5, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 5, 0, 0, 0, 0, 5, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 5, 0, 0, 0, 0, 5, 3],
], dtype=int)

E3_IN = np.array([
    [0, 5, 0, 0, 5, 0, 5, 0, 5, 0],
    [0, 5, 0, 0, 5, 0, 5, 0, 5, 0],
    [0, 5, 0, 0, 5, 0, 5, 0, 5, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 5, 0, 0, 5, 0, 5, 0, 5, 0],
    [0, 5, 0, 0, 5, 0, 5, 0, 5, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 5, 0, 0, 5, 0, 5, 0, 5, 0],
    [0, 5, 0, 0, 5, 0, 5, 0, 5, 0],
    [0, 5, 0, 0, 5, 0, 5, 0, 5, 0],
], dtype=int)

E3_OUT = np.array([
    [1, 5, 0, 0, 5, 0, 5, 0, 5, 0],
    [1, 5, 0, 0, 5, 0, 5, 0, 5, 0],
    [1, 5, 0, 0, 5, 0, 5, 0, 5, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 5, 0, 0, 5, 2, 5, 0, 5, 0],
    [0, 5, 0, 0, 5, 2, 5, 0, 5, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 5, 0, 0, 5, 0, 5, 0, 5, 3],
    [0, 5, 0, 0, 5, 0, 5, 0, 5, 3],
    [0, 5, 0, 0, 5, 0, 5, 0, 5, 3],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 5, 0, 5, 0, 0, 5, 0, 5, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 5, 0, 5, 0, 0, 5, 0, 5, 0],
    [0, 5, 0, 5, 0, 0, 5, 0, 5, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 5, 0, 5, 0, 0, 5, 0, 5, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 5, 0, 5, 0, 0, 5, 0, 5, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 5, 0, 5, 0, 0, 5, 0, 5, 0],
], dtype=int)

T_OUT = np.array([
    [1, 5, 0, 5, 0, 0, 5, 0, 5, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 5, 0, 5, 0, 0, 5, 0, 5, 0],
    [0, 5, 0, 5, 0, 0, 5, 0, 5, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 5, 0, 5, 2, 2, 5, 0, 5, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 5, 0, 5, 0, 0, 5, 0, 5, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 5, 0, 5, 0, 0, 5, 0, 5, 3],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def f(j,A,c,E):
 if not(0<=A<len(j)and 0<=c<len(j[0])):return
 if j[A][c]:return
 j[A][c]=E
 for k,W in[(0,-1),(0,1),(-1,0),(1,0)]:f(j,A+k,c+W,E)
def p(j):
 l,J=len(j),len(j[0]);f(j,0,0,1)
 for a in range(4):f(j,l//2-1+a%2,J//2-1+a//2,2)
 f(j,l-1,J-1,3);return j


# --- Code Golf Solution (Compressed) ---
def q(g, k=7, r=range(10)):
    return g * -k or p([(q := 0) or [(q := (g[i][~j] or [~i // 4 * -(j ^ 9 - i < 2 & ~j), q % 5][k < 7])) for i in r] for j in r], k - 1)


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import randint, sample, uniform

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

Piece = Union[Grid, Patch]

F = False

T = True

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

def argmin(
    container: Container,
    compfunc: Callable
) -> Any:
    """ smallest item by custom order """
    return min(container, key=compfunc, default=None)

def sfilter(
    container: Container,
    condition: Callable
) -> Container:
    """ keep elements in container that satisfy condition """
    return type(container)(e for e in container if condition(e))

def extract(
    container: Container,
    condition: Callable
) -> Any:
    """ first element of container that satisfies condition """
    return next(e for e in container if condition(e))

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

def canvas(
    value: Integer,
    dimensions: IntegerTuple
) -> Grid:
    """ grid construction """
    return tuple(tuple(value for j in range(dimensions[1])) for i in range(dimensions[0]))

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

def generate_941d9a10(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2, 3))
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    opts = interval(2, (h-1)//2 + 1, 2)
    nhidx = unifint(diff_lb, diff_ub, (0, len(opts) - 1))
    nh = opts[nhidx]
    opts = interval(2, (w-1)//2 + 1, 2)
    nwidx = unifint(diff_lb, diff_ub, (0, len(opts) - 1))
    nw = opts[nwidx]
    bgc, fgc = sample(cols, 2)
    hgrid = canvas(bgc, (2*nh+1, w))
    for j in range(1, h, 2):
        hgrid = fill(hgrid, fgc, connect((j, 0), (j, w)))
    for k in range(h - (2*nh+1)):
        loc = randint(0, height(hgrid) - 1)
        hgrid = hgrid[:loc] + canvas(bgc, (1, w)) + hgrid[loc:]
    wgrid = canvas(bgc, (2*nw+1, h))
    for j in range(1, w, 2):
        wgrid = fill(wgrid, fgc, connect((j, 0), (j, h)))
    for k in range(w - (2*nw+1)):
        loc = randint(0, height(wgrid) - 1)
        wgrid = wgrid[:loc] + canvas(bgc, (1, h)) + wgrid[loc:]
    wgrid = dmirror(wgrid)
    gi = canvas(bgc, (h, w))
    fronts = ofcolor(hgrid, fgc) | ofcolor(wgrid, fgc)
    gi = fill(gi, fgc, fronts)
    objs = objects(gi, T, T, F)
    objs = colorfilter(objs, bgc)
    blue = argmin(objs, lambda o: leftmost(o) + uppermost(o))
    green = argmax(objs, lambda o: leftmost(o) + uppermost(o))
    f1 = lambda o: len(sfilter(objs, lambda o2: leftmost(o2) < leftmost(o))) == len(sfilter(objs, lambda o2: leftmost(o2) > leftmost(o)))
    f2 = lambda o: len(sfilter(objs, lambda o2: uppermost(o2) < uppermost(o))) == len(sfilter(objs, lambda o2: uppermost(o2) > uppermost(o)))
    red = extract(objs, lambda o: f1(o) and f2(o))
    go = fill(gi, 1, blue)
    go = fill(go, 3, green)
    go = fill(go, 2, red)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Numerical = Union[Integer, IntegerTuple]

ONE = 1

TWO = 2

THREE = 3

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

def equality(
    a: Any,
    b: Any
) -> Boolean:
    """ equality """
    return a == b

def greater(
    a: Integer,
    b: Integer
) -> Boolean:
    """ greater """
    return a > b

def size(
    container: Container
) -> Integer:
    """ cardinality """
    return len(container)

def both(
    a: Boolean,
    b: Boolean
) -> Boolean:
    """ logical and """
    return a and b

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

def toobject(
    patch: Patch,
    grid: Grid
) -> Object:
    """ object from patch and grid """
    h, w = len(grid), len(grid[0])
    return frozenset((grid[i][j], (i, j)) for i, j in toindices(patch) if 0 <= i < h and 0 <= j < w)

def corners(
    patch: Patch
) -> Indices:
    """ indices of corners """
    return frozenset({ulcorner(patch), urcorner(patch), llcorner(patch), lrcorner(patch)})

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_941d9a10(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = asindices(I)
    x1 = corners(x0)
    x2 = toobject(x1, I)
    x3 = mostcolor(x2)
    x4 = objects(I, T, T, F)
    x5 = colorfilter(x4, x3)
    x6 = fork(add, leftmost, uppermost)
    x7 = argmin(x5, x6)
    x8 = argmax(x5, x6)
    x9 = lbind(sfilter, x5)
    x10 = rbind(compose, leftmost)
    x11 = chain(size, x9, x10)
    x12 = lbind(sfilter, x5)
    x13 = rbind(compose, uppermost)
    x14 = chain(size, x12, x13)
    x15 = lbind(lbind, greater)
    x16 = chain(x11, x15, leftmost)
    x17 = lbind(rbind, greater)
    x18 = chain(x11, x17, leftmost)
    x19 = lbind(lbind, greater)
    x20 = chain(x14, x19, uppermost)
    x21 = lbind(rbind, greater)
    x22 = chain(x14, x21, uppermost)
    x23 = fork(equality, x16, x18)
    x24 = fork(equality, x20, x22)
    x25 = fork(both, x23, x24)
    x26 = extract(x5, x25)
    x27 = fill(I, ONE, x7)
    x28 = fill(x27, THREE, x8)
    x29 = fill(x28, TWO, x26)
    return x29


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_941d9a10(inp)
        assert pred == _to_grid(expected), f"{name} failed"

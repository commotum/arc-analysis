# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "9af7a82c"
SERIAL = "239"
URL    = "https://arcprize.org/play?task=9af7a82c"

# --- Code Golf Concepts ---
CONCEPTS = [
    "separate_images",
    "count_tiles",
    "summarize",
    "order_numbers",
]

# --- Example Grids ---
E1_IN = np.array([
    [2, 2, 1],
    [2, 3, 1],
    [1, 1, 1],
], dtype=int)

E1_OUT = np.array([
    [1, 2, 3],
    [1, 2, 0],
    [1, 2, 0],
    [1, 0, 0],
    [1, 0, 0],
], dtype=int)

E2_IN = np.array([
    [3, 1, 1, 4],
    [2, 2, 2, 4],
    [4, 4, 4, 4],
], dtype=int)

E2_OUT = np.array([
    [4, 2, 1, 3],
    [4, 2, 1, 0],
    [4, 2, 0, 0],
    [4, 0, 0, 0],
    [4, 0, 0, 0],
    [4, 0, 0, 0],
], dtype=int)

E3_IN = np.array([
    [8, 8, 2],
    [3, 8, 8],
    [3, 3, 4],
    [3, 3, 4],
], dtype=int)

E3_OUT = np.array([
    [3, 8, 4, 2],
    [3, 8, 4, 0],
    [3, 8, 0, 0],
    [3, 8, 0, 0],
    [3, 0, 0, 0],
], dtype=int)

E4_IN = np.array([
    [1, 1, 1],
    [2, 2, 1],
    [2, 8, 1],
    [2, 8, 1],
], dtype=int)

E4_OUT = np.array([
    [1, 2, 8],
    [1, 2, 8],
    [1, 2, 0],
    [1, 2, 0],
    [1, 0, 0],
    [1, 0, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [8, 8, 2, 2],
    [1, 8, 8, 2],
    [1, 3, 3, 4],
    [1, 1, 1, 1],
], dtype=int)

T_OUT = np.array([
    [1, 8, 2, 3, 4],
    [1, 8, 2, 3, 0],
    [1, 8, 2, 0, 0],
    [1, 8, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
from collections import*
def p(j,A=range):
 c=Counter([x for r in j for x in r]).most_common(9);E,k=c[0][1],len(c);j=[[0 for _ in A(k)]for _ in A(E)]
 for W in A(k):
  for l in A(c[W][1]):j[l][W]=c[W][0]
 return j


# --- Code Golf Solution (Compressed) ---
def q(g):
    return [(s := sum(g, [])), *filter(any, zip(*sorted(([(c := (-s.count(e)))] + [e] * -c + [0] * 99 for e in {*s}))))][2:]


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import sample, uniform

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

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

def generate_9af7a82c(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(1, 10, 1)
    prods = dict()
    for a in range(1, 31, 1):
        for b in range(1, 31, 1):
            prd = a*b
            if prd in prods:
                prods[prd].append((a, b))
            else:
                prods[prd] = [(a, b)]
    ncols = unifint(diff_lb, diff_ub, (2, 9))
    leastnc = sum(range(1, ncols + 1, 1))
    maxnc = sum(range(30, 30 - ncols, -1))
    cands = {k: v for k, v in prods.items() if leastnc <= k <= maxnc}
    options = set()
    for v in cands.values():
        for opt in v:
            options.add(opt)
    options = sorted(options, key=lambda ij: ij[0] * ij[1])
    idx = unifint(diff_lb, diff_ub, (0, len(options) - 1))
    h, w = options[idx]
    ccols = sample(cols, ncols)
    counts = list(range(1, ncols + 1, 1))
    eliginds = {ncols - 1}
    while sum(counts) < h * w:
        eligindss = sorted(eliginds, reverse=True)
        idx = unifint(diff_lb, diff_ub, (0, len(eligindss) - 1))
        idx = eligindss[idx]
        counts[idx] += 1
        if idx > 0:
            eliginds.add(idx - 1)
        if idx < ncols - 1:
            if counts[idx] == counts[idx+1] - 1:
                eliginds = eliginds - {idx}
        if counts[idx] == 30:
            eliginds = eliginds - {idx}
    gi = canvas(-1, (h, w))
    go = canvas(0, (max(counts), ncols))
    inds = asindices(gi)
    counts = counts[::-1]
    for j, (col, cnt) in enumerate(zip(ccols, counts)):
        locs = sample(totuple(inds), cnt)
        gi = fill(gi, col, locs)
        inds = inds - set(locs)
        go = fill(go, col, connect((0, j), (cnt - 1, j)))
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Numerical = Union[Integer, IntegerTuple]

IntegerSet = FrozenSet[Integer]

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

ContainerContainer = Container[Container]

ZERO = 0

ONE = 1

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

def order(
    container: Container,
    compfunc: Callable
) -> Tuple:
    """ order container by custom key """
    return tuple(sorted(container, key=compfunc))

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

def partition(
    grid: Grid
) -> Objects:
    """ each cell with the same value part of the same object """
    return frozenset(
        frozenset(
            (v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value
        ) for value in palette(grid)
    )

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

def vmirror(
    piece: Piece
) -> Piece:
    """ mirroring along vertical """
    if isinstance(piece, tuple):
        return tuple(row[::-1] for row in piece)
    d = ulcorner(piece)[1] + lrcorner(piece)[1]
    if isinstance(next(iter(piece))[1], tuple):
        return frozenset((v, (i, d - j)) for v, (i, j) in piece)
    return frozenset((i, d - j) for i, j in piece)

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

def cmirror(
    piece: Piece
) -> Piece:
    """ mirroring along counterdiagonal """
    if isinstance(piece, tuple):
        return tuple(zip(*(r[::-1] for r in piece[::-1])))
    return vmirror(dmirror(vmirror(piece)))

def vconcat(
    a: Grid,
    b: Grid
) -> Grid:
    """ concatenate two grids vertically """
    return a + b

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_9af7a82c(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = partition(I)
    x1 = order(x0, size)
    x2 = valmax(x0, size)
    x3 = rbind(astuple, ONE)
    x4 = lbind(subtract, x2)
    x5 = compose(x3, size)
    x6 = chain(x3, x4, size)
    x7 = fork(canvas, color, x5)
    x8 = lbind(canvas, ZERO)
    x9 = compose(x8, x6)
    x10 = fork(vconcat, x7, x9)
    x11 = compose(cmirror, x10)
    x12 = apply(x11, x1)
    x13 = merge(x12)
    x14 = cmirror(x13)
    return x14


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("E4", E4_IN, E4_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_9af7a82c(inp)
        assert pred == _to_grid(expected), f"{name} failed"

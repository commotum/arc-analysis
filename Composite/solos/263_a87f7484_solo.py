# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "a87f7484"
SERIAL = "263"
URL    = "https://arcprize.org/play?task=a87f7484"

# --- Code Golf Concepts ---
CONCEPTS = [
    "separate_images",
    "find_the_intruder",
    "crop",
]

# --- Example Grids ---
E1_IN = np.array([
    [6, 0, 6],
    [0, 6, 6],
    [6, 0, 6],
    [4, 0, 4],
    [0, 4, 4],
    [4, 0, 4],
    [8, 8, 8],
    [8, 0, 8],
    [8, 8, 8],
], dtype=int)

E1_OUT = np.array([
    [8, 8, 8],
    [8, 0, 8],
    [8, 8, 8],
], dtype=int)

E2_IN = np.array([
    [2, 0, 0, 3, 0, 0, 7, 0, 7, 1, 0, 0],
    [2, 0, 0, 3, 0, 0, 0, 7, 0, 1, 0, 0],
    [0, 2, 2, 0, 3, 3, 7, 0, 7, 0, 1, 1],
], dtype=int)

E2_OUT = np.array([
    [7, 0, 7],
    [0, 7, 0],
    [7, 0, 7],
], dtype=int)

E3_IN = np.array([
    [3, 0, 0, 4, 0, 4, 2, 0, 0, 8, 0, 0, 1, 0, 0],
    [0, 3, 3, 4, 4, 4, 0, 2, 2, 0, 8, 8, 0, 1, 1],
    [0, 3, 0, 4, 0, 4, 0, 2, 0, 0, 8, 0, 0, 1, 0],
], dtype=int)

E3_OUT = np.array([
    [4, 0, 4],
    [4, 4, 4],
    [4, 0, 4],
], dtype=int)

E4_IN = np.array([
    [0, 7, 7],
    [7, 7, 0],
    [7, 0, 7],
    [3, 0, 0],
    [0, 3, 3],
    [3, 0, 0],
    [2, 0, 0],
    [0, 2, 2],
    [2, 0, 0],
    [8, 0, 0],
    [0, 8, 8],
    [8, 0, 0],
], dtype=int)

E4_OUT = np.array([
    [0, 7, 7],
    [7, 7, 0],
    [7, 0, 7],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 5, 0],
    [5, 0, 5],
    [0, 5, 0],
    [0, 3, 0],
    [3, 0, 3],
    [0, 3, 0],
    [6, 0, 6],
    [6, 6, 0],
    [6, 0, 6],
    [0, 4, 0],
    [4, 0, 4],
    [0, 4, 0],
    [0, 8, 0],
    [8, 0, 8],
    [0, 8, 0],
], dtype=int)

T_OUT = np.array([
    [6, 0, 6],
    [6, 6, 0],
    [6, 0, 6],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
	A=range;c=[[[j[D+c*3][A+E*3]for A in A(3)]for D in A(3)]for c in A(len(j)//3)for E in A(len(j[0])//3)]
	for E in c:
		if[tuple(tuple(c[E][A]==0 for A in A(3))for E in A(3))for c in c].count(tuple(tuple(E[c][A]==0 for A in A(3))for c in A(3)))==1:return E


# --- Code Golf Solution (Compressed) ---
def q(g, h=0):
    return -(M := bytes(map(bool, sum((g := [*zip(*(h or p(g, g)))]), ())))).find(M[:9], 9) * g[:3] or p(g[3:] + g[:3])


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, sample, shuffle, uniform

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

def totuple(
    container: FrozenSet
) -> Tuple:
    """ conversion to tuple """
    return tuple(container)

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

def toindices(
    patch: Patch
) -> Indices:
    """ indices of object cells """
    if len(patch) == 0:
        return frozenset()
    if isinstance(next(iter(patch))[1], tuple):
        return frozenset(index for value, index in patch)
    return patch

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

def generate_a87f7484(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 5))
    w = unifint(diff_lb, diff_ub, (2, 30))
    num = unifint(diff_lb, diff_ub, (3, min(30 // h, 9)))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    ccols = sample(remcols, num)
    ncd = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    nc = choice((ncd, h * w - ncd))
    nc = min(max(1, nc), h * w - 1)
    c = canvas(bgc, (h, w))
    inds = asindices(c)
    origlocs = sample(totuple(inds), nc)
    canbrem = {l for l in origlocs}
    canbeadd = inds - set(origlocs)
    otherlocs = {l for l in origlocs}
    nchangesinv = unifint(diff_lb, diff_ub, (0, h * w - 1))
    nchanges = h * w - nchangesinv
    for k in range(nchanges):
        if choice((True, False)):
            if len(canbrem) > 1:
                ch = choice(totuple(canbrem))
                otherlocs = remove(ch, otherlocs)
                canbrem = remove(ch, canbrem)
            elif len(canbeadd) > 1:
                ch = choice(totuple(canbeadd))
                otherlocs = insert(ch, otherlocs)
                canbeadd = remove(ch, canbeadd)
        else:
            if len(canbeadd) > 1:
                ch = choice(totuple(canbeadd))
                otherlocs = insert(ch, otherlocs)
                canbeadd = remove(ch, canbeadd)
            elif len(canbrem) > 1:
                ch = choice(totuple(canbrem))
                otherlocs = remove(ch, otherlocs)
                canbrem = remove(ch, canbrem)
    go = fill(c, ccols[0], origlocs)
    grids = [go]
    for cc in ccols[1:]:
        grids.append(fill(c, cc, otherlocs))
    shuffle(grids)
    grids = tuple(grids)
    gi = merge(grids)
    if choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Numerical = Union[Integer, IntegerTuple]

IntegerSet = FrozenSet[Integer]

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

ONE = 1

TWO = 2

THREE = 3

TEN = 10

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

def halve(
    n: Numerical
) -> Numerical:
    """ scaling by one half """
    return n // 2 if isinstance(n, int) else (n[0] // 2, n[1] // 2)

def equality(
    a: Any,
    b: Any
) -> Boolean:
    """ equality """
    return a == b

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

def dedupe(
    iterable: Tuple
) -> Tuple:
    """ remove duplicates """
    return tuple(e for i, e in enumerate(iterable) if iterable.index(e) == i)

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

def argmin(
    container: Container,
    compfunc: Callable
) -> Any:
    """ smallest item by custom order """
    return min(container, key=compfunc, default=None)

def mostcommon(
    container: Container
) -> Any:
    """ most common item """
    return max(set(container), key=container.count)

def increment(
    x: Numerical
) -> Numerical:
    """ incrementing """
    return x + 1 if isinstance(x, int) else (x[0] + 1, x[1] + 1)

def positive(
    x: Integer
) -> Boolean:
    """ positive """
    return x > 0

def sfilter(
    container: Container,
    condition: Callable
) -> Container:
    """ keep elements in container that satisfy condition """
    return type(container)(e for e in container if condition(e))

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

def apply(
    function: Callable,
    container: Container
) -> Container:
    """ apply function to each item in container """
    return type(container)(function(e) for e in container)

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

def crop(
    grid: Grid,
    start: IntegerTuple,
    dims: IntegerTuple
) -> Grid:
    """ subgrid specified by start and dimension """
    return tuple(r[start[1]:start[1]+dims[1]] for r in grid[start[0]:start[0]+dims[0]])

def partition(
    grid: Grid
) -> Objects:
    """ each cell with the same value part of the same object """
    return frozenset(
        frozenset(
            (v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value
        ) for value in palette(grid)
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

def hsplit(
    grid: Grid,
    n: Integer
) -> Tuple:
    """ split grid horizontally """
    h, w = len(grid), len(grid[0]) // n
    offset = len(grid[0]) % n != 0
    return tuple(crop(grid, (0, w * i + i * offset), (h, w)) for i in range(n))

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

def verify_a87f7484(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = height(I)
    x1 = halve(x0)
    x2 = increment(x1)
    x3 = interval(THREE, x2, ONE)
    x4 = width(I)
    x5 = halve(x4)
    x6 = increment(x5)
    x7 = interval(THREE, x6, ONE)
    x8 = palette(I)
    x9 = lbind(apply, toindices)
    x10 = compose(x9, partition)
    x11 = rbind(compose, palette)
    x12 = lbind(lbind, contained)
    x13 = compose(x11, x12)
    x14 = lbind(chain, size)
    x15 = rbind(x14, x13)
    x16 = lbind(lbind, sfilter)
    x17 = compose(x15, x16)
    x18 = compose(positive, size)
    x19 = lbind(sfilter, x8)
    x20 = fork(matcher, x17, size)
    x21 = chain(x18, x19, x20)
    x22 = lbind(apply, shape)
    x23 = chain(size, dedupe, x22)
    x24 = matcher(x23, ONE)
    x25 = lbind(apply, x10)
    x26 = chain(size, dedupe, x25)
    x27 = matcher(x26, TWO)
    x28 = compose(size, dedupe)
    x29 = fork(equality, size, x28)
    x30 = fork(add, x21, x24)
    x31 = fork(add, x27, x29)
    x32 = fork(add, x30, x31)
    x33 = multiply(TEN, TEN)
    x34 = lbind(multiply, x33)
    x35 = compose(x34, x32)
    x36 = fork(add, x35, size)
    x37 = lbind(vsplit, I)
    x38 = apply(x37, x3)
    x39 = lbind(hsplit, I)
    x40 = apply(x39, x7)
    x41 = combine(x38, x40)
    x42 = argmax(x41, x36)
    x43 = apply(x10, x42)
    x44 = mostcommon(x43)
    x45 = matcher(x10, x44)
    x46 = argmin(x42, x45)
    return x46


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("E4", E4_IN, E4_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_a87f7484(inp)
        assert pred == _to_grid(expected), f"{name} failed"

# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "3f7978a0"
SERIAL = "091"
URL    = "https://arcprize.org/play?task=3f7978a0"

# --- Code Golf Concepts ---
CONCEPTS = [
    "crop",
    "rectangle_guessing",
    "find_the_intruder",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 8, 0, 0, 0, 8, 0, 0, 8],
    [0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 5, 0, 8, 0, 5, 0, 8, 0],
    [0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 8, 0, 0, 0, 8, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 8, 0],
    [0, 8, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [8, 0, 0, 0, 8],
    [5, 0, 0, 0, 5],
    [5, 0, 8, 0, 5],
    [5, 0, 0, 0, 5],
    [8, 0, 0, 0, 8],
], dtype=int)

E2_IN = np.array([
    [0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8],
    [0, 0, 8, 0, 0, 0, 0, 0, 8, 0, 0],
    [8, 0, 5, 0, 0, 0, 0, 0, 5, 0, 0],
    [0, 0, 5, 0, 0, 8, 8, 0, 5, 0, 0],
    [0, 0, 5, 0, 0, 0, 0, 0, 5, 0, 0],
    [0, 0, 8, 0, 8, 0, 0, 0, 8, 0, 0],
    [0, 8, 0, 0, 0, 0, 0, 0, 8, 8, 0],
], dtype=int)

E2_OUT = np.array([
    [8, 0, 0, 0, 0, 0, 8],
    [5, 0, 0, 0, 0, 0, 5],
    [5, 0, 0, 8, 8, 0, 5],
    [5, 0, 0, 0, 0, 0, 5],
    [8, 0, 8, 0, 0, 0, 8],
], dtype=int)

E3_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 8, 0, 0, 0],
    [0, 0, 8, 5, 0, 8, 0, 5, 0, 0, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 8, 0, 0, 0, 0],
    [0, 0, 8, 5, 0, 8, 0, 5, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 8, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0],
    [0, 0, 8, 8, 0, 0, 0, 8, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [8, 0, 0, 0, 8],
    [5, 0, 0, 0, 5],
    [5, 0, 8, 0, 5],
    [5, 0, 0, 0, 5],
    [5, 0, 8, 0, 5],
    [8, 0, 0, 0, 8],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [8, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [5, 8, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 8],
    [5, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [5, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [5, 0, 0, 5, 0, 0, 8, 0, 0, 8, 0, 0, 0],
    [5, 0, 8, 5, 8, 0, 0, 0, 0, 0, 0, 0, 8],
    [5, 0, 0, 5, 0, 8, 0, 0, 0, 0, 0, 0, 0],
    [5, 8, 0, 5, 0, 0, 0, 0, 0, 0, 8, 0, 8],
    [5, 0, 0, 5, 0, 0, 0, 8, 0, 0, 0, 0, 0],
    [8, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
], dtype=int)

T_OUT = np.array([
    [8, 0, 0, 8],
    [5, 8, 0, 5],
    [5, 0, 0, 5],
    [5, 0, 0, 5],
    [5, 0, 0, 5],
    [5, 0, 8, 5],
    [5, 0, 0, 5],
    [5, 8, 0, 5],
    [5, 0, 0, 5],
    [8, 0, 0, 8],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
	A=len;c=range;E=[]
	for k in c(A(j[0])):
		if any(j[c][k]==5 for c in c(A(j))):E.append(k)
	W=[]
	for l in c(A(j)):
		if j[l][E[0]]==5:W.append(l)
	J,a=min(W)-1,max(W)+1;C,e=E[0],E[1];return[[j[E][c]for c in c(C,e+1)]for E in c(J,a+1)]


# --- Code Golf Solution (Compressed) ---
def q(g, k=46):
    return ~k * g or p([*zip(*g[(5 in g[k | -2]) - 2::-1])], k - 1)


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

def crop(
    grid: Grid,
    start: IntegerTuple,
    dims: IntegerTuple
) -> Grid:
    """ subgrid specified by start and dimension """
    return tuple(r[start[1]:start[1]+dims[1]] for r in grid[start[0]:start[0]+dims[0]])

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

def subgrid(
    patch: Patch,
    grid: Grid
) -> Grid:
    """ smallest subgrid containing object """
    return crop(grid, ulcorner(patch), shape(patch))

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

def generate_3f7978a0(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    bgc, noisec, linec = sample(cols, 3)
    c = canvas(bgc, (h, w))
    oh = unifint(diff_lb, diff_ub, (4, max(4, int((2/3) * h))))
    oh = min(oh, h)
    ow = unifint(diff_lb, diff_ub, (4, max(4, int((2/3) * w))))
    ow = min(ow, w)
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    nnoise = unifint(diff_lb, diff_ub, (0, (h * w) // 4))
    inds = totuple(asindices(c))
    noise = sample(inds, nnoise)
    gi = fill(c, noisec, noise)
    ulc = (loci, locj)
    lrc = (loci + oh - 1, locj + ow - 1)
    llc = (loci + oh - 1, locj)
    urc = (loci, locj + ow - 1)
    gi = fill(gi, linec, connect(ulc, llc))
    gi = fill(gi, linec, connect(urc, lrc))
    crns = {ulc, lrc, llc, urc}
    gi = fill(gi, noisec, crns)
    go = subgrid(crns, gi)
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

TWO = 2

F = False

T = True

DOWN = (1, 0)

RIGHT = (0, 1)

UP = (-1, 0)

LEFT = (0, -1)

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

def double(
    n: Numerical
) -> Numerical:
    """ scaling by two """
    return n * 2 if isinstance(n, int) else (n[0] * 2, n[1] * 2)

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

def initset(
    value: Any
) -> FrozenSet:
    """ initialize container """
    return frozenset({value})

def both(
    a: Boolean,
    b: Boolean
) -> Boolean:
    """ logical and """
    return a and b

def either(
    a: Boolean,
    b: Boolean
) -> Boolean:
    """ logical or """
    return a or b

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

def lrcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of lower right corner """
    return tuple(map(max, zip(*toindices(patch))))

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

def partition(
    grid: Grid
) -> Objects:
    """ each cell with the same value part of the same object """
    return frozenset(
        frozenset(
            (v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value
        ) for value in palette(grid)
    )

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

def color(
    obj: Object
) -> Integer:
    """ color of object """
    return next(iter(obj))[0]

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_3f7978a0(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = partition(I)
    x1 = objects(I, T, F, F)
    x2 = compose(double, height)
    x3 = fork(equality, x2, size)
    x4 = compose(double, width)
    x5 = fork(equality, x4, size)
    x6 = fork(either, x3, x5)
    x7 = rbind(equality, TWO)
    x8 = lbind(colorfilter, x1)
    x9 = rbind(sfilter, vline)
    x10 = rbind(sfilter, hline)
    x11 = chain(x9, x8, color)
    x12 = chain(x7, size, x11)
    x13 = chain(x10, x8, color)
    x14 = chain(x7, size, x13)
    x15 = fork(either, x12, x14)
    x16 = fork(both, x6, x15)
    x17 = extract(x0, x16)
    x18 = color(x17)
    x19 = colorfilter(x1, x18)
    x20 = first(x19)
    x21 = vline(x20)
    x22 = ulcorner(x17)
    x23 = lrcorner(x17)
    x24 = branch(x21, UP, LEFT)
    x25 = add(x22, x24)
    x26 = branch(x21, DOWN, RIGHT)
    x27 = add(x23, x26)
    x28 = initset(x27)
    x29 = insert(x25, x28)
    x30 = subgrid(x29, I)
    return x30


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_3f7978a0(inp)
        assert pred == _to_grid(expected), f"{name} failed"

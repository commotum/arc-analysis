# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "85c4e7cd"
SERIAL = "203"
URL    = "https://arcprize.org/play?task=85c4e7cd"

# --- Code Golf Concepts ---
CONCEPTS = [
    "color_guessing",
    "recoloring",
    "color_permutation",
]

# --- Example Grids ---
E1_IN = np.array([
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4],
    [4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4],
    [4, 2, 1, 3, 3, 3, 3, 3, 3, 1, 2, 4],
    [4, 2, 1, 3, 5, 5, 5, 5, 3, 1, 2, 4],
    [4, 2, 1, 3, 5, 8, 8, 5, 3, 1, 2, 4],
    [4, 2, 1, 3, 5, 8, 8, 5, 3, 1, 2, 4],
    [4, 2, 1, 3, 5, 5, 5, 5, 3, 1, 2, 4],
    [4, 2, 1, 3, 3, 3, 3, 3, 3, 1, 2, 4],
    [4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4],
    [4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
], dtype=int)

E1_OUT = np.array([
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8],
    [8, 5, 3, 3, 3, 3, 3, 3, 3, 3, 5, 8],
    [8, 5, 3, 1, 1, 1, 1, 1, 1, 3, 5, 8],
    [8, 5, 3, 1, 2, 2, 2, 2, 1, 3, 5, 8],
    [8, 5, 3, 1, 2, 4, 4, 2, 1, 3, 5, 8],
    [8, 5, 3, 1, 2, 4, 4, 2, 1, 3, 5, 8],
    [8, 5, 3, 1, 2, 2, 2, 2, 1, 3, 5, 8],
    [8, 5, 3, 1, 1, 1, 1, 1, 1, 3, 5, 8],
    [8, 5, 3, 3, 3, 3, 3, 3, 3, 3, 5, 8],
    [8, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
], dtype=int)

E2_IN = np.array([
    [2, 2, 2, 2, 2, 2],
    [2, 1, 1, 1, 1, 2],
    [2, 1, 6, 6, 1, 2],
    [2, 1, 6, 6, 1, 2],
    [2, 1, 1, 1, 1, 2],
    [2, 2, 2, 2, 2, 2],
], dtype=int)

E2_OUT = np.array([
    [6, 6, 6, 6, 6, 6],
    [6, 1, 1, 1, 1, 6],
    [6, 1, 2, 2, 1, 6],
    [6, 1, 2, 2, 1, 6],
    [6, 1, 1, 1, 1, 6],
    [6, 6, 6, 6, 6, 6],
], dtype=int)

E3_IN = np.array([
    [8, 8, 8, 8, 8, 8, 8, 8],
    [8, 1, 1, 1, 1, 1, 1, 8],
    [8, 1, 2, 2, 2, 2, 1, 8],
    [8, 1, 2, 4, 4, 2, 1, 8],
    [8, 1, 2, 4, 4, 2, 1, 8],
    [8, 1, 2, 2, 2, 2, 1, 8],
    [8, 1, 1, 1, 1, 1, 1, 8],
    [8, 8, 8, 8, 8, 8, 8, 8],
], dtype=int)

E3_OUT = np.array([
    [4, 4, 4, 4, 4, 4, 4, 4],
    [4, 2, 2, 2, 2, 2, 2, 4],
    [4, 2, 1, 1, 1, 1, 2, 4],
    [4, 2, 1, 8, 8, 1, 2, 4],
    [4, 2, 1, 8, 8, 1, 2, 4],
    [4, 2, 1, 1, 1, 1, 2, 4],
    [4, 2, 2, 2, 2, 2, 2, 4],
    [4, 4, 4, 4, 4, 4, 4, 4],
], dtype=int)

E4_IN = np.array([
    [7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
    [7, 2, 2, 2, 2, 2, 2, 2, 2, 7],
    [7, 2, 4, 4, 4, 4, 4, 4, 2, 7],
    [7, 2, 4, 1, 1, 1, 1, 4, 2, 7],
    [7, 2, 4, 1, 3, 3, 1, 4, 2, 7],
    [7, 2, 4, 1, 3, 3, 1, 4, 2, 7],
    [7, 2, 4, 1, 1, 1, 1, 4, 2, 7],
    [7, 2, 4, 4, 4, 4, 4, 4, 2, 7],
    [7, 2, 2, 2, 2, 2, 2, 2, 2, 7],
    [7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
], dtype=int)

E4_OUT = np.array([
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 1, 1, 1, 1, 1, 1, 1, 1, 3],
    [3, 1, 4, 4, 4, 4, 4, 4, 1, 3],
    [3, 1, 4, 2, 2, 2, 2, 4, 1, 3],
    [3, 1, 4, 2, 7, 7, 2, 4, 1, 3],
    [3, 1, 4, 2, 7, 7, 2, 4, 1, 3],
    [3, 1, 4, 2, 2, 2, 2, 4, 1, 3],
    [3, 1, 4, 4, 4, 4, 4, 4, 1, 3],
    [3, 1, 1, 1, 1, 1, 1, 1, 1, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 8],
    [8, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 8],
    [8, 2, 4, 3, 3, 3, 3, 3, 3, 3, 3, 4, 2, 8],
    [8, 2, 4, 3, 7, 7, 7, 7, 7, 7, 3, 4, 2, 8],
    [8, 2, 4, 3, 7, 6, 6, 6, 6, 7, 3, 4, 2, 8],
    [8, 2, 4, 3, 7, 6, 5, 5, 6, 7, 3, 4, 2, 8],
    [8, 2, 4, 3, 7, 6, 5, 5, 6, 7, 3, 4, 2, 8],
    [8, 2, 4, 3, 7, 6, 6, 6, 6, 7, 3, 4, 2, 8],
    [8, 2, 4, 3, 7, 7, 7, 7, 7, 7, 3, 4, 2, 8],
    [8, 2, 4, 3, 3, 3, 3, 3, 3, 3, 3, 4, 2, 8],
    [8, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 8],
    [8, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
], dtype=int)

T_OUT = np.array([
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5],
    [5, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 5],
    [5, 6, 7, 3, 3, 3, 3, 3, 3, 3, 3, 7, 6, 5],
    [5, 6, 7, 3, 4, 4, 4, 4, 4, 4, 3, 7, 6, 5],
    [5, 6, 7, 3, 4, 2, 2, 2, 2, 4, 3, 7, 6, 5],
    [5, 6, 7, 3, 4, 2, 8, 8, 2, 4, 3, 7, 6, 5],
    [5, 6, 7, 3, 4, 2, 8, 8, 2, 4, 3, 7, 6, 5],
    [5, 6, 7, 3, 4, 2, 2, 2, 2, 4, 3, 7, 6, 5],
    [5, 6, 7, 3, 4, 4, 4, 4, 4, 4, 3, 7, 6, 5],
    [5, 6, 7, 3, 3, 3, 3, 3, 3, 3, 3, 7, 6, 5],
    [5, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 5],
    [5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(g,L=len,R=range):
 h=L(g)
 w=L(g[0])
 C=g[h//2][:w//2]
 C={C[i]:C[-(i+1)] for i in R(L(C))}
 for r in R(h):
  for c in R(w):g[r][c]=C[g[r][c]]
 return g


# --- Code Golf Solution (Compressed) ---
def q(g):
    return [[g[(n := (len(g) // 2))][r.index(c) + n] for c in r] for r in g]


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, sample, uniform

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

def interval(
    start: Integer,
    stop: Integer,
    step: Integer
) -> Tuple:
    """ range """
    return tuple(range(start, stop, step))

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

def generate_85c4e7cd(diff_lb: float, diff_ub: float) -> dict:
    colopts = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 15))
    w = unifint(diff_lb, diff_ub, (1, 15))
    ncols = unifint(diff_lb, diff_ub, (1, 10))
    cols = sample(colopts, ncols)
    colord = [choice(cols) for j in range(min(h, w))]
    shp = (h*2, w*2)
    gi = canvas(0, shp)
    go = canvas(0, shp)
    for idx, (ci, co) in enumerate(zip(colord, colord[::-1])):
        ulc = (idx, idx)
        lrc = (h*2 - 1 - idx, w*2 - 1 - idx)
        bx = box(frozenset({ulc, lrc}))
        gi = fill(gi, ci, bx)
        go = fill(go, co, bx)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Numerical = Union[Integer, IntegerTuple]

IntegerSet = FrozenSet[Integer]

Piece = Union[Grid, Patch]

TupleTuple = Tuple[Tuple]

ContainerContainer = Container[Container]

ZERO = 0

ONE = 1

def invert(
    n: Numerical
) -> Numerical:
    """ inversion with respect to addition """
    return -n if isinstance(n, int) else (-n[0], -n[1])

def halve(
    n: Numerical
) -> Numerical:
    """ scaling by one half """
    return n // 2 if isinstance(n, int) else (n[0] // 2, n[1] // 2)

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

def repeat(
    item: Any,
    num: Integer
) -> Tuple:
    """ repetition of item within vector """
    return tuple(item for i in range(num))

def merge(
    containers: ContainerContainer
) -> Container:
    """ merging """
    return type(containers)(e for c in containers for e in c)

def minimum(
    container: IntegerSet
) -> Integer:
    """ minimum """
    return min(container, default=0)

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

def last(
    container: Container
) -> Any:
    """ last item of container """
    return max(enumerate(container))[1]

def pair(
    a: Tuple,
    b: Tuple
) -> TupleTuple:
    """ zipping of two tuples """
    return tuple(zip(a, b))

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

def power(
    function: Callable,
    n: Integer
) -> Callable:
    """ power of function """
    if n == 1:
        return function
    return compose(function, power(function, n - 1))

def apply(
    function: Callable,
    container: Container
) -> Container:
    """ apply function to each item in container """
    return type(container)(function(e) for e in container)

def rapply(
    functions: Container,
    value: Any
) -> Container:
    """ apply each function in container to value """
    return type(functions)(function(value) for function in functions)

def papply(
    function: Callable,
    a: Tuple,
    b: Tuple
) -> Tuple:
    """ apply function on two vectors """
    return tuple(function(i, j) for i, j in zip(a, b))

def mpapply(
    function: Callable,
    a: Tuple,
    b: Tuple
) -> Tuple:
    """ apply function on two vectors and merge """
    return merge(papply(function, a, b))

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

def recolor(
    value: Integer,
    patch: Patch
) -> Object:
    """ recolor patch """
    return frozenset((value, index) for index in toindices(patch))

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

def inbox(
    patch: Patch
) -> Indices:
    """ inbox for patch """
    ai, aj = uppermost(patch) + 1, leftmost(patch) + 1
    bi, bj = lowermost(patch) - 1, rightmost(patch) - 1
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

def verify_85c4e7cd(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = asindices(I)
    x1 = box(x0)
    x2 = shape(I)
    x3 = minimum(x2)
    x4 = halve(x3)
    x5 = interval(ONE, x4, ONE)
    x6 = lbind(power, inbox)
    x7 = rbind(rapply, x1)
    x8 = compose(initset, x6)
    x9 = chain(first, x7, x8)
    x10 = apply(x9, x5)
    x11 = repeat(x1, ONE)
    x12 = combine(x11, x10)
    x13 = rbind(toobject, I)
    x14 = compose(color, x13)
    x15 = apply(x14, x12)
    x16 = interval(ZERO, x4, ONE)
    x17 = pair(x16, x15)
    x18 = compose(invert, first)
    x19 = order(x17, x18)
    x20 = apply(last, x19)
    x21 = mpapply(recolor, x20, x12)
    x22 = paint(I, x21)
    return x22


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("E4", E4_IN, E4_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_85c4e7cd(inp)
        assert pred == _to_grid(expected), f"{name} failed"

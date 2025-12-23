# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "27a28665"
SERIAL = "056"
URL    = "https://arcprize.org/play?task=27a28665"

# --- Code Golf Concepts ---
CONCEPTS = [
    "associate_colors_to_patterns",
    "take_negative",
    "associate_images_to_patterns",
]

# --- Example Grids ---
E1_IN = np.array([
    [5, 5, 0],
    [5, 0, 5],
    [0, 5, 0],
], dtype=int)

E1_OUT = np.array([
    [1],
], dtype=int)

E2_IN = np.array([
    [8, 0, 8],
    [0, 8, 0],
    [8, 0, 8],
], dtype=int)

E2_OUT = np.array([
    [2],
], dtype=int)

E3_IN = np.array([
    [5, 0, 5],
    [0, 5, 0],
    [5, 0, 5],
], dtype=int)

E3_OUT = np.array([
    [2],
], dtype=int)

E4_IN = np.array([
    [0, 1, 1],
    [0, 1, 1],
    [1, 0, 0],
], dtype=int)

E4_OUT = np.array([
    [3],
], dtype=int)

E5_IN = np.array([
    [0, 8, 8],
    [0, 8, 8],
    [8, 0, 0],
], dtype=int)

E5_OUT = np.array([
    [3],
], dtype=int)

E6_IN = np.array([
    [4, 4, 0],
    [4, 0, 4],
    [0, 4, 0],
], dtype=int)

E6_OUT = np.array([
    [1],
], dtype=int)

E7_IN = np.array([
    [0, 5, 0],
    [5, 5, 5],
    [0, 5, 0],
], dtype=int)

E7_OUT = np.array([
    [6],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 8, 0],
    [8, 8, 8],
    [0, 8, 0],
], dtype=int)

T_OUT = np.array([
    [6],
], dtype=int)

T2_IN = np.array([
    [7, 7, 0],
    [7, 0, 7],
    [0, 7, 0],
], dtype=int)

T2_OUT = np.array([
    [1],
], dtype=int)

T3_IN = np.array([
    [2, 0, 2],
    [0, 2, 0],
    [2, 0, 2],
], dtype=int)

T3_OUT = np.array([
    [2],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):A=tuple(0if v==0else 1for v in j[0]);return[[{(1,1,0):1,(1,0,1):2,(0,1,1):3,(0,1,0):6}[A]]]


# --- Code Golf Solution (Compressed) ---
def q(g):
    return [[2 ^ (g[2] < [1]) * 3 + (g < [[1]])]]


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

Element = Union[Object, Grid]

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

def asobject(
    grid: Grid
) -> Object:
    """ conversion of grid to object """
    return frozenset((v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r))

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

def upscale(
    element: Element,
    factor: Integer
) -> Element:
    """ upscale object or grid """
    if isinstance(element, tuple):
        upscaled_grid = tuple()
        for row in element:
            upscaled_row = tuple()
            for value in row:
                upscaled_row = upscaled_row + tuple(value for num in range(factor))
            upscaled_grid = upscaled_grid + tuple(upscaled_row for num in range(factor))
        return upscaled_grid
    else:
        if len(element) == 0:
            return frozenset()
        di_inv, dj_inv = ulcorner(element)
        di, dj = (-di_inv, -dj_inv)
        normed_obj = shift(element, (di, dj))
        upscaled_obj = set()
        for value, (i, j) in normed_obj:
            for io in range(factor):
                for jo in range(factor):
                    upscaled_obj.add((value, (i * factor + io, j * factor + jo)))
        return shift(frozenset(upscaled_obj), (di_inv, dj_inv))

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

def generate_27a28665(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    mapping = [
    (1, {(0, 0), (0, 1), (1, 0), (1, 2), (2, 1)}),
    (2, {(0, 0), (1, 1), (2, 0), (0, 2), (2, 2)}),
    (3, {(2, 0), (0, 1), (0, 2), (1, 1), (1, 2)}),
    (6, {(1, 1), (0, 1), (1, 0), (1, 2), (2, 1)})
    ]
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    col, obj = choice(mapping)
    bgc, objc = sample(cols, 2)
    fac = unifint(diff_lb, diff_ub, (1, min(h, w) // 3))
    go = canvas(col, (1, 1))
    gi = canvas(bgc, (h, w))
    canv = canvas(bgc, (3, 3))
    canv = fill(canv, objc, obj)
    canv = upscale(canv, fac)
    obj = asobject(canv)
    loci = randint(0, h - 3 * fac)
    locj = randint(0, w - 3 * fac)
    loc = (loci, locj)
    gi = paint(gi, shift(obj, loc))
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Numerical = Union[Integer, IntegerTuple]

Objects = FrozenSet[Object]

Piece = Union[Grid, Patch]

TupleTuple = Tuple[Tuple]

ZERO = 0

ONE = 1

TWO = 2

THREE = 3

FOUR = 4

FIVE = 5

SIX = 6

TEN = 10

F = False

T = True

UNITY = (1, 1)

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

def divide(
    a: Numerical,
    b: Numerical
) -> Numerical:
    """ floor division """
    if isinstance(a, int) and isinstance(b, int):
        return a // b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] // b[0], a[1] // b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a // b[0], a // b[1])
    return (a[0] // b, a[1] // b)

def equality(
    a: Any,
    b: Any
) -> Boolean:
    """ equality """
    return a == b

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

def valmax(
    container: Container,
    compfunc: Callable
) -> Integer:
    """ maximum by custom function """
    return compfunc(max(container, key=compfunc, default=0))

def initset(
    value: Any
) -> FrozenSet:
    """ initialize container """
    return frozenset({value})

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

def rapply(
    functions: Container,
    value: Any
) -> Container:
    """ apply each function in container to value """
    return type(functions)(function(value) for function in functions)

def mostcolor(
    element: Element
) -> Integer:
    """ most common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return max(set(values), key=values.count)

def width(
    piece: Piece
) -> Integer:
    """ width of grid or patch """
    if len(piece) == 0:
        return 0
    if isinstance(piece, tuple):
        return len(piece[0])
    return rightmost(piece) - leftmost(piece) + 1

def asindices(
    grid: Grid
) -> Indices:
    """ indices of all grid cells """
    return frozenset((i, j) for i in range(len(grid)) for j in range(len(grid[0])))

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

def downscale(
    grid: Grid,
    factor: Integer
) -> Grid:
    """ downscale grid """
    h, w = len(grid), len(grid[0])
    downscaled_grid = tuple()
    for i in range(h):
        downscaled_row = tuple()
        for j in range(w):
            if j % factor == 0:
                downscaled_row = downscaled_row + (grid[i][j],)
        downscaled_grid = downscaled_grid + (downscaled_row, )
    h = len(downscaled_grid)
    downscaled_grid2 = tuple()
    for i in range(h):
        if i % factor == 0:
            downscaled_grid2 = downscaled_grid2 + (downscaled_grid[i],)
    return downscaled_grid2

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_27a28665(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = lbind(apply, last)
    x1 = compose(positive, first)
    x2 = lbind(interval, ZERO)
    x3 = rbind(x2, ONE)
    x4 = rbind(sfilter, x1)
    x5 = compose(x3, size)
    x6 = fork(pair, x5, identity)
    x7 = chain(x0, x4, x6)
    x8 = rbind(branch, identity)
    x9 = rbind(x8, x7)
    x10 = chain(size, dedupe, first)
    x11 = lbind(equality, ONE)
    x12 = chain(x9, x11, x10)
    x13 = compose(initset, x12)
    x14 = fork(rapply, x13, identity)
    x15 = compose(first, x14)
    x16 = rbind(branch, identity)
    x17 = rbind(x16, x15)
    x18 = chain(x17, positive, size)
    x19 = compose(initset, x18)
    x20 = fork(rapply, x19, identity)
    x21 = compose(first, x20)
    x22 = multiply(TEN, THREE)
    x23 = power(x21, x22)
    x24 = compose(rot90, x23)
    x25 = power(x24, FOUR)
    x26 = x25(I)
    x27 = width(x26)
    x28 = divide(x27, THREE)
    x29 = downscale(x26, x28)
    x30 = objects(x29, T, F, F)
    x31 = valmax(x30, size)
    x32 = equality(x31, ONE)
    x33 = equality(x31, FOUR)
    x34 = equality(x31, FIVE)
    x35 = branch(x32, TWO, ONE)
    x36 = branch(x33, THREE, x35)
    x37 = branch(x34, SIX, x36)
    x38 = canvas(x37, UNITY)
    return x38


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("E4", E4_IN, E4_OUT),
        ("E5", E5_IN, E5_OUT),
        ("E6", E6_IN, E6_OUT),
        ("E7", E7_IN, E7_OUT),
        ("T", T_IN, T_OUT),
        ("T2", T2_IN, T2_OUT),
        ("T3", T3_IN, T3_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_27a28665(inp)
        assert pred == _to_grid(expected), f"{name} failed"

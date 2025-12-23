# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "995c5fa3"
SERIAL = "235"
URL    = "https://arcprize.org/play?task=995c5fa3"

# --- Code Golf Concepts ---
CONCEPTS = [
    "take_complement",
    "detect_wall",
    "separate_images",
    "associate_colors_to_images",
    "summarize",
]

# --- Example Grids ---
E1_IN = np.array([
    [5, 5, 5, 5, 0, 5, 5, 5, 5, 0, 5, 5, 5, 5],
    [5, 5, 5, 5, 0, 5, 0, 0, 5, 0, 0, 5, 5, 0],
    [5, 5, 5, 5, 0, 5, 0, 0, 5, 0, 0, 5, 5, 0],
    [5, 5, 5, 5, 0, 5, 5, 5, 5, 0, 5, 5, 5, 5],
], dtype=int)

E1_OUT = np.array([
    [2, 2, 2],
    [8, 8, 8],
    [3, 3, 3],
], dtype=int)

E2_IN = np.array([
    [5, 5, 5, 5, 0, 5, 5, 5, 5, 0, 5, 5, 5, 5],
    [0, 5, 5, 0, 0, 5, 5, 5, 5, 0, 5, 5, 5, 5],
    [0, 5, 5, 0, 0, 5, 0, 0, 5, 0, 5, 5, 5, 5],
    [5, 5, 5, 5, 0, 5, 0, 0, 5, 0, 5, 5, 5, 5],
], dtype=int)

E2_OUT = np.array([
    [3, 3, 3],
    [4, 4, 4],
    [2, 2, 2],
], dtype=int)

E3_IN = np.array([
    [5, 5, 5, 5, 0, 5, 5, 5, 5, 0, 5, 5, 5, 5],
    [5, 0, 0, 5, 0, 5, 5, 5, 5, 0, 5, 5, 5, 5],
    [5, 0, 0, 5, 0, 5, 5, 5, 5, 0, 5, 0, 0, 5],
    [5, 5, 5, 5, 0, 5, 5, 5, 5, 0, 5, 0, 0, 5],
], dtype=int)

E3_OUT = np.array([
    [8, 8, 8],
    [2, 2, 2],
    [4, 4, 4],
], dtype=int)

E4_IN = np.array([
    [5, 5, 5, 5, 0, 5, 5, 5, 5, 0, 5, 5, 5, 5],
    [5, 5, 5, 5, 0, 5, 5, 5, 5, 0, 5, 5, 5, 5],
    [5, 5, 5, 5, 0, 5, 0, 0, 5, 0, 5, 5, 5, 5],
    [5, 5, 5, 5, 0, 5, 0, 0, 5, 0, 5, 5, 5, 5],
], dtype=int)

E4_OUT = np.array([
    [2, 2, 2],
    [4, 4, 4],
    [2, 2, 2],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [5, 5, 5, 5, 0, 5, 5, 5, 5, 0, 5, 5, 5, 5],
    [5, 5, 5, 5, 0, 0, 5, 5, 0, 0, 5, 0, 0, 5],
    [5, 0, 0, 5, 0, 0, 5, 5, 0, 0, 5, 0, 0, 5],
    [5, 0, 0, 5, 0, 5, 5, 5, 5, 0, 5, 5, 5, 5],
], dtype=int)

T_OUT = np.array([
    [4, 4, 4],
    [3, 3, 3],
    [8, 8, 8],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
    return [[(45 - j[2][x] - 2 * j[2][x + 1] - 4 * j[1][x + 1]) // 5] * 3 for x in range(0, 15, 5)]


# --- Code Golf Solution (Compressed) ---
def q(g):
    return [[g[1][x] * sum(g[2][x:x + 3]) % 13 ^ 8] * 3 for x in b'\x01\x06\x0b']


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, uniform

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

def repeat(
    item: Any,
    num: Integer
) -> Tuple:
    """ repetition of item within vector """
    return tuple(item for i in range(num))

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

def generate_995c5fa3(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    o1 = asindices(canvas(-1, (4, 4)))
    o2 = box(asindices(canvas(-1, (4, 4))))
    o3 = asindices(canvas(-1, (4, 4))) - {(1, 0), (2, 0), (1, 3), (2, 3)}
    o4 = o1 - shift(asindices(canvas(-1, (2, 2))), (2, 1))
    mpr = [(o1, 2), (o2, 8), (o3, 3), (o4, 4)]
    num = unifint(diff_lb, diff_ub, (1, 6))
    h = 4
    w = 4 * num + num - 1
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    ccols = []
    for k in range(num):
        col = choice(remcols)
        obj, outcol = choice(mpr)
        locj = 5 * k
        gi = fill(gi, col, shift(obj, (0, locj)))
        ccols.append(outcol)
    go = tuple(repeat(c, num) for c in ccols)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Numerical = Union[Integer, IntegerTuple]

IntegerSet = FrozenSet[Integer]

Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

ZERO = 0

ONE = 1

TWO = 2

THREE = 3

FOUR = 4

FIVE = 5

EIGHT = 8

NEG_ONE = -1

DOWN = (1, 0)

def identity(
    x: Any
) -> Any:
    """ identity function """
    return x

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

def either(
    a: Boolean,
    b: Boolean
) -> Boolean:
    """ logical or """
    return a or b

def increment(
    x: Numerical
) -> Numerical:
    """ incrementing """
    return x + 1 if isinstance(x, int) else (x[0] + 1, x[1] + 1)

def tojvec(
    j: Integer
) -> IntegerTuple:
    """ vector pointing horizontally """
    return (0, j)

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

def width(
    piece: Piece
) -> Integer:
    """ width of grid or patch """
    if len(piece) == 0:
        return 0
    if isinstance(piece, tuple):
        return len(piece[0])
    return rightmost(piece) - leftmost(piece) + 1

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

def numcolors(
    element: Element
) -> IntegerSet:
    """ number of colors occurring in object or grid """
    return len(palette(element))

def toobject(
    patch: Patch,
    grid: Grid
) -> Object:
    """ object from patch and grid """
    h, w = len(grid), len(grid[0])
    return frozenset((grid[i][j], (i, j)) for i, j in toindices(patch) if 0 <= i < h and 0 <= j < w)

def hmirror(
    piece: Piece
) -> Piece:
    """ mirroring along horizontal """
    if isinstance(piece, tuple):
        return piece[::-1]
    d = ulcorner(piece)[0] + lrcorner(piece)[0]
    if isinstance(next(iter(piece))[1], tuple):
        return frozenset((v, (d - i, j)) for v, (i, j) in piece)
    return frozenset((d - i, j) for i, j in piece)

def index(
    grid: Grid,
    loc: IntegerTuple
) -> Integer:
    """ color at location """
    i, j = loc
    h, w = len(grid), len(grid[0])
    if not (0 <= i < h and 0 <= j < w):
        return None
    return grid[loc[0]][loc[1]]

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_995c5fa3(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = width(I)
    x1 = increment(x0)
    x2 = divide(x1, FIVE)
    x3 = astuple(FOUR, FOUR)
    x4 = canvas(NEG_ONE, x3)
    x5 = asindices(x4)
    x6 = rbind(toobject, I)
    x7 = lbind(shift, x5)
    x8 = compose(x6, x7)
    x9 = multiply(x2, FIVE)
    x10 = interval(ZERO, x9, FIVE)
    x11 = apply(tojvec, x10)
    x12 = apply(x8, x11)
    x13 = matcher(numcolors, ONE)
    x14 = fork(equality, identity, hmirror)
    x15 = compose(flip, x14)
    x16 = lbind(index, I)
    x17 = compose(x16, ulcorner)
    x18 = lbind(add, DOWN)
    x19 = chain(x16, x18, ulcorner)
    x20 = fork(equality, x17, x19)
    x21 = compose(flip, x20)
    x22 = fork(either, x13, x15)
    x23 = fork(either, x22, x21)
    x24 = compose(flip, x23)
    x25 = lbind(multiply, TWO)
    x26 = compose(x25, x13)
    x27 = lbind(multiply, FOUR)
    x28 = compose(x27, x15)
    x29 = fork(add, x26, x28)
    x30 = lbind(multiply, THREE)
    x31 = compose(x30, x21)
    x32 = lbind(multiply, EIGHT)
    x33 = compose(x32, x24)
    x34 = fork(add, x31, x33)
    x35 = fork(add, x29, x34)
    x36 = apply(x35, x12)
    x37 = rbind(repeat, x2)
    x38 = apply(x37, x36)
    return x38


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("E4", E4_IN, E4_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_995c5fa3(inp)
        assert pred == _to_grid(expected), f"{name} failed"

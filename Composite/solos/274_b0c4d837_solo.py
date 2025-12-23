# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "b0c4d837"
SERIAL = "274"
URL    = "https://arcprize.org/play?task=b0c4d837"

# --- Code Golf Concepts ---
CONCEPTS = [
    "measure_length",
    "associate_images_to_numbers",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 5, 0, 0, 5, 0],
    [0, 5, 0, 0, 5, 0],
    [0, 5, 0, 0, 5, 0],
    [0, 5, 8, 8, 5, 0],
    [0, 5, 5, 5, 5, 0],
], dtype=int)

E1_OUT = np.array([
    [8, 8, 8],
    [0, 0, 0],
    [0, 0, 0],
], dtype=int)

E2_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 5, 0, 0, 0, 5, 0, 0],
    [0, 0, 5, 0, 0, 0, 5, 0, 0],
    [0, 0, 5, 0, 0, 0, 5, 0, 0],
    [0, 0, 5, 0, 0, 0, 5, 0, 0],
    [0, 0, 5, 8, 8, 8, 5, 0, 0],
    [0, 0, 5, 8, 8, 8, 5, 0, 0],
    [0, 0, 5, 8, 8, 8, 5, 0, 0],
    [0, 0, 5, 5, 5, 5, 5, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [8, 8, 8],
    [0, 0, 8],
    [0, 0, 0],
], dtype=int)

E3_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 5, 0, 0, 0, 0, 0, 5, 0],
    [0, 5, 0, 0, 0, 0, 0, 5, 0],
    [0, 5, 0, 0, 0, 0, 0, 5, 0],
    [0, 5, 8, 8, 8, 8, 8, 5, 0],
    [0, 5, 8, 8, 8, 8, 8, 5, 0],
    [0, 5, 8, 8, 8, 8, 8, 5, 0],
    [0, 5, 5, 5, 5, 5, 5, 5, 0],
], dtype=int)

E3_OUT = np.array([
    [8, 8, 8],
    [0, 0, 0],
    [0, 0, 0],
], dtype=int)

E4_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 5, 0, 0, 0, 5, 0, 0],
    [0, 0, 5, 0, 0, 0, 5, 0, 0],
    [0, 0, 5, 8, 8, 8, 5, 0, 0],
    [0, 0, 5, 8, 8, 8, 5, 0, 0],
    [0, 0, 5, 8, 8, 8, 5, 0, 0],
    [0, 0, 5, 8, 8, 8, 5, 0, 0],
    [0, 0, 5, 5, 5, 5, 5, 0, 0],
], dtype=int)

E4_OUT = np.array([
    [8, 8, 0],
    [0, 0, 0],
    [0, 0, 0],
], dtype=int)

E5_IN = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 5, 0, 0, 5, 0],
    [0, 5, 8, 8, 5, 0],
    [0, 5, 8, 8, 5, 0],
    [0, 5, 5, 5, 5, 0],
], dtype=int)

E5_OUT = np.array([
    [8, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
], dtype=int)

E6_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 5, 0, 0, 0, 5, 0],
    [0, 5, 0, 0, 0, 5, 0],
    [0, 5, 8, 8, 8, 5, 0],
    [0, 5, 8, 8, 8, 5, 0],
    [0, 5, 5, 5, 5, 5, 0],
], dtype=int)

E6_OUT = np.array([
    [8, 8, 0],
    [0, 0, 0],
    [0, 0, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 5, 0, 0, 0, 5, 0, 0],
    [0, 0, 5, 8, 8, 8, 5, 0, 0],
    [0, 0, 5, 8, 8, 8, 5, 0, 0],
    [0, 0, 5, 8, 8, 8, 5, 0, 0],
    [0, 0, 5, 8, 8, 8, 5, 0, 0],
    [0, 0, 5, 8, 8, 8, 5, 0, 0],
    [0, 0, 5, 8, 8, 8, 5, 0, 0],
    [0, 0, 5, 5, 5, 5, 5, 0, 0],
], dtype=int)

T_OUT = np.array([
    [8, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
j=lambda A,c:sum(sum(i==c for i in r)for r in A)
def p(A):E=max(j([r],8)for r in A);k=(j(A,5)-E-2)/2-j(A,8)/E;return[[8*(k>0),8*(k>1),8*(k>2)],[0,0,8*(k>3)],[0,0,0]]


# --- Code Golf Solution (Compressed) ---
def q(g):
    return [[8, (s := (sum(map(max, g)) * 6)) + 4 & 8, -s & 8], [0, 0, ~s & 8], [0] * 3]


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

def identity(
    x: Any
) -> Any:
    """ identity function """
    return x

def repeat(
    item: Any,
    num: Integer
) -> Tuple:
    """ repetition of item within vector """
    return tuple(item for i in range(num))

def interval(
    start: Integer,
    stop: Integer,
    step: Integer
) -> Tuple:
    """ range """
    return tuple(range(start, stop, step))

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

def rot90(
    grid: Grid
) -> Grid:
    """ quarter clockwise rotation """
    return tuple(row for row in zip(*grid[::-1]))

def rot180(
    grid: Grid
) -> Grid:
    """ half rotation """
    return tuple(tuple(row[::-1]) for row in grid[::-1])

def rot270(
    grid: Grid
) -> Grid:
    """ quarter anticlockwise rotation """
    return tuple(tuple(row[::-1]) for row in zip(*grid[::-1]))[::-1]

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

def generate_b0c4d837(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    oh = unifint(diff_lb, diff_ub, (3, h - 1))
    ow = unifint(diff_lb, diff_ub, (3, w - 1))
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    bgc, boxc, fillc = sample(cols, 3)
    subg = canvas(boxc, (oh, ow))
    subg2 = canvas(fillc, (oh-1, ow-2))
    ntofill = unifint(diff_lb, diff_ub, (1, min(9, oh-2)))
    for j in range(ntofill):
        subg2 = fill(subg2, bgc, connect((j, 0), (j, ow-2)))
    subg = paint(subg, shift(asobject(subg2), (0, 1)))
    gi = canvas(bgc, (h, w))
    gi = paint(gi, shift(asobject(subg), (loci, locj)))
    go = repeat(fillc, ntofill) + repeat(bgc, 9 - ntofill)
    go = (go[:3], go[3:6][::-1], go[6:])
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Numerical = Union[Integer, IntegerTuple]

IntegerSet = FrozenSet[Integer]

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

TupleTuple = Tuple[Tuple]

ZERO = 0

ONE = 1

TWO = 2

THREE = 3

NINE = 9

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

def equality(
    a: Any,
    b: Any
) -> Boolean:
    """ equality """
    return a == b

def combine(
    a: Container,
    b: Container
) -> Container:
    """ union """
    return type(a)((*a, *b))

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

def argmin(
    container: Container,
    compfunc: Callable
) -> Any:
    """ smallest item by custom order """
    return min(container, key=compfunc, default=None)

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

def remove(
    value: Any,
    container: Container
) -> Container:
    """ remove item from container """
    return type(container)(e for e in container if e != value)

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

def matcher(
    function: Callable,
    target: Any
) -> Callable:
    """ construction of equality function """
    return lambda x: function(x) == target

def fork(
    outer: Callable,
    a: Callable,
    b: Callable
) -> Callable:
    """ creates a wrapper function """
    return lambda x: outer(a(x), b(x))

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

def vconcat(
    a: Grid,
    b: Grid
) -> Grid:
    """ concatenate two grids vertically """
    return a + b

def hsplit(
    grid: Grid,
    n: Integer
) -> Tuple:
    """ split grid horizontally """
    h, w = len(grid), len(grid[0]) // n
    offset = len(grid[0]) % n != 0
    return tuple(crop(grid, (0, w * i + i * offset), (h, w)) for i in range(n))

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_b0c4d837(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = partition(I)
    x1 = fork(multiply, height, width)
    x2 = argmax(x0, x1)
    x3 = remove(x2, x0)
    x4 = argmin(x3, x1)
    x5 = argmax(x3, x1)
    x6 = ulcorner(x5)
    x7 = llcorner(x5)
    x8 = connect(x6, x7)
    x9 = urcorner(x5)
    x10 = lrcorner(x5)
    x11 = connect(x9, x10)
    x12 = combine(x8, x11)
    x13 = toindices(x5)
    x14 = difference(x12, x13)
    x15 = size(x14)
    x16 = equality(x15, ZERO)
    x17 = branch(x16, height, width)
    x18 = x17(x5)
    x19 = x17(x4)
    x20 = subtract(x18, x19)
    x21 = decrement(x20)
    x22 = color(x4)
    x23 = color(x2)
    x24 = repeat(x22, x21)
    x25 = subtract(NINE, x21)
    x26 = repeat(x23, x25)
    x27 = combine(x24, x26)
    x28 = repeat(x27, ONE)
    x29 = hsplit(x28, THREE)
    x30 = interval(ZERO, THREE, ONE)
    x31 = pair(x30, x29)
    x32 = matcher(first, ZERO)
    x33 = extract(x31, x32)
    x34 = last(x33)
    x35 = matcher(first, ONE)
    x36 = extract(x31, x35)
    x37 = last(x36)
    x38 = matcher(first, TWO)
    x39 = extract(x31, x38)
    x40 = last(x39)
    x41 = vmirror(x37)
    x42 = vconcat(x34, x41)
    x43 = vconcat(x42, x40)
    return x43


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("E4", E4_IN, E4_OUT),
        ("E5", E5_IN, E5_OUT),
        ("E6", E6_IN, E6_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_b0c4d837(inp)
        assert pred == _to_grid(expected), f"{name} failed"

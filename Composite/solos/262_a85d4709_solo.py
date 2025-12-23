# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "a85d4709"
SERIAL = "262"
URL    = "https://arcprize.org/play?task=a85d4709"

# --- Code Golf Concepts ---
CONCEPTS = [
    "separate_images",
    "associate_colors_to_images",
    "summarize",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 5],
    [0, 5, 0],
    [5, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [3, 3, 3],
    [4, 4, 4],
    [2, 2, 2],
], dtype=int)

E2_IN = np.array([
    [0, 0, 5],
    [0, 0, 5],
    [0, 0, 5],
], dtype=int)

E2_OUT = np.array([
    [3, 3, 3],
    [3, 3, 3],
    [3, 3, 3],
], dtype=int)

E3_IN = np.array([
    [5, 0, 0],
    [0, 5, 0],
    [5, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [2, 2, 2],
    [4, 4, 4],
    [2, 2, 2],
], dtype=int)

E4_IN = np.array([
    [0, 5, 0],
    [0, 0, 5],
    [0, 5, 0],
], dtype=int)

E4_OUT = np.array([
    [4, 4, 4],
    [3, 3, 3],
    [4, 4, 4],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 5],
    [5, 0, 0],
    [0, 5, 0],
], dtype=int)

T_OUT = np.array([
    [3, 3, 3],
    [2, 2, 2],
    [4, 4, 4],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
    return [[[2, 4, 3][r.index(5)]] * 3 for r in j]


# --- Code Golf Solution (Compressed) ---
def q(m):
    return [3 * [b % 4 + ~a % 4] for a, b, c in m]


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

def difference(
    a: Container,
    b: Container
) -> Container:
    """ difference """
    return type(a)(e for e in a if e not in b)

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

def generate_a85d4709(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (2, 3, 4))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w3 = unifint(diff_lb, diff_ub, (1, 10))
    w = w3 * 3
    bgc, dotc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    for ii in range(h):
        loc = randint(0, w3 - 1)
        dev = unifint(diff_lb, diff_ub, (0, w3 // 2 + 1))
        loc = w3 // 3 + choice((+dev, -dev))
        loc = min(max(0, loc), w3 - 1)
        ofs, col = choice(((0, 2), (1, 4), (2, 3)))
        loc += ofs * w3
        gi = fill(gi, dotc, {(ii, loc)})
        ln = connect((ii, 0), (ii, w - 1))
        go = fill(go, col, ln)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Numerical = Union[Integer, IntegerTuple]

Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

TWO = 2

THREE = 3

FOUR = 4

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

def repeat(
    item: Any,
    num: Integer
) -> Tuple:
    """ repetition of item within vector """
    return tuple(item for i in range(num))

def greater(
    a: Integer,
    b: Integer
) -> Boolean:
    """ greater """
    return a > b

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

def compose(
    outer: Callable,
    inner: Callable
) -> Callable:
    """ function composition """
    return lambda x: outer(inner(x))

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

def leastcolor(
    element: Element
) -> Integer:
    """ least common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return min(set(values), key=values.count)

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

def verify_a85d4709(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = leastcolor(I)
    x1 = height(I)
    x2 = vsplit(I, x1)
    x3 = rbind(ofcolor, x0)
    x4 = compose(leftmost, x3)
    x5 = width(I)
    x6 = divide(x5, THREE)
    x7 = multiply(x6, TWO)
    x8 = lbind(greater, x6)
    x9 = compose(x8, x4)
    x10 = lbind(greater, x7)
    x11 = compose(x10, x4)
    x12 = compose(flip, x9)
    x13 = fork(both, x11, x12)
    x14 = fork(either, x9, x13)
    x15 = compose(flip, x14)
    x16 = rbind(multiply, TWO)
    x17 = compose(x16, x9)
    x18 = rbind(multiply, FOUR)
    x19 = compose(x18, x13)
    x20 = rbind(multiply, THREE)
    x21 = compose(x20, x15)
    x22 = fork(add, x17, x19)
    x23 = fork(add, x22, x21)
    x24 = width(I)
    x25 = rbind(repeat, x24)
    x26 = compose(x25, x23)
    x27 = apply(x26, x2)
    return x27


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("E4", E4_IN, E4_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_a85d4709(inp)
        assert pred == _to_grid(expected), f"{name} failed"

# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "e9afcf9a"
SERIAL = "373"
URL    = "https://arcprize.org/play?task=e9afcf9a"

# --- Code Golf Concepts ---
CONCEPTS = [
    "pattern_modification",
]

# --- Example Grids ---
E1_IN = np.array([
    [3, 3, 3, 3, 3, 3],
    [9, 9, 9, 9, 9, 9],
], dtype=int)

E1_OUT = np.array([
    [3, 9, 3, 9, 3, 9],
    [9, 3, 9, 3, 9, 3],
], dtype=int)

E2_IN = np.array([
    [4, 4, 4, 4, 4, 4],
    [8, 8, 8, 8, 8, 8],
], dtype=int)

E2_OUT = np.array([
    [4, 8, 4, 8, 4, 8],
    [8, 4, 8, 4, 8, 4],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [6, 6, 6, 6, 6, 6],
    [2, 2, 2, 2, 2, 2],
], dtype=int)

T_OUT = np.array([
    [6, 2, 6, 2, 6, 2],
    [2, 6, 2, 6, 2, 6],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(g):
    return [[[g[i][j], g[1 - i][j]][j % 2] for j in range(6)] for i in range(2)]


# --- Code Golf Solution (Compressed) ---
def q(g):
    return [(T := (max(zip(*g)) * 3)), T[::-1]]


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

Piece = Union[Grid, Patch]

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

def generate_e9afcf9a(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    numc = unifint(diff_lb, diff_ub, (1, min(10, h)))
    colss = sample(cols, numc)
    rr = tuple(choice(colss) for k in range(h))
    rr2 = rr[::-1]
    gi = []
    go = []
    for k in range(w):
        gi.append(rr)
        if k % 2 == 0:
            go.append(rr)
        else:
            go.append(rr2)
    gi = dmirror(tuple(gi))
    go = dmirror(tuple(go))
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Numerical = Union[Integer, IntegerTuple]

def identity(
    x: Any
) -> Any:
    """ identity function """
    return x

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

def double(
    n: Numerical
) -> Numerical:
    """ scaling by two """
    return n * 2 if isinstance(n, int) else (n[0] * 2, n[1] * 2)

def halve(
    n: Numerical
) -> Numerical:
    """ scaling by one half """
    return n // 2 if isinstance(n, int) else (n[0] // 2, n[1] // 2)

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

def decrement(
    x: Numerical
) -> Numerical:
    """ decrementing """
    return x - 1 if isinstance(x, int) else (x[0] - 1, x[1] - 1)

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

def asobject(
    grid: Grid
) -> Object:
    """ conversion of grid to object """
    return frozenset((v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r))

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

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_e9afcf9a(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = asobject(I)
    x1 = height(I)
    x2 = decrement(x1)
    x3 = lbind(subtract, x2)
    x4 = compose(double, halve)
    x5 = fork(equality, identity, x4)
    x6 = compose(last, last)
    x7 = chain(flip, x5, x6)
    x8 = sfilter(x0, x7)
    x9 = chain(x3, first, last)
    x10 = compose(last, last)
    x11 = fork(astuple, x9, x10)
    x12 = fork(astuple, first, x11)
    x13 = apply(x12, x8)
    x14 = paint(I, x13)
    return x14


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_e9afcf9a(inp)
        assert pred == _to_grid(expected), f"{name} failed"

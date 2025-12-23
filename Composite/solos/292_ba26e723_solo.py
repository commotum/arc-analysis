# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "ba26e723"
SERIAL = "292"
URL    = "https://arcprize.org/play?task=ba26e723"

# --- Code Golf Concepts ---
CONCEPTS = [
    "pattern_modification",
    "pairwise_analogy",
    "recoloring",
]

# --- Example Grids ---
E1_IN = np.array([
    [4, 0, 4, 0, 4, 0, 4, 0, 4, 0],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [0, 4, 0, 4, 0, 4, 0, 4, 0, 4],
], dtype=int)

E1_OUT = np.array([
    [6, 0, 4, 0, 4, 0, 6, 0, 4, 0],
    [6, 4, 4, 6, 4, 4, 6, 4, 4, 6],
    [0, 4, 0, 6, 0, 4, 0, 4, 0, 6],
], dtype=int)

E2_IN = np.array([
    [0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4],
], dtype=int)

E2_OUT = np.array([
    [0, 4, 0, 6, 0, 4, 0, 4, 0, 6, 0],
    [6, 4, 4, 6, 4, 4, 6, 4, 4, 6, 4],
    [6, 0, 4, 0, 4, 0, 6, 0, 4, 0, 4],
], dtype=int)

E3_IN = np.array([
    [4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0],
], dtype=int)

E3_OUT = np.array([
    [6, 0, 4, 0, 4, 0, 6, 0, 4, 0, 4],
    [6, 4, 4, 6, 4, 4, 6, 4, 4, 6, 4],
    [0, 4, 0, 6, 0, 4, 0, 4, 0, 6, 0],
], dtype=int)

E4_IN = np.array([
    [4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0],
], dtype=int)

E4_OUT = np.array([
    [6, 0, 4, 0, 4, 0, 6, 0, 4, 0, 4, 0, 6],
    [6, 4, 4, 6, 4, 4, 6, 4, 4, 6, 4, 4, 6],
    [0, 4, 0, 6, 0, 4, 0, 4, 0, 6, 0, 4, 0],
], dtype=int)

E5_IN = np.array([
    [0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0],
], dtype=int)

E5_OUT = np.array([
    [0, 4, 0, 6, 0, 4, 0, 4, 0, 6, 0, 4, 0, 4],
    [6, 4, 4, 6, 4, 4, 6, 4, 4, 6, 4, 4, 6, 4],
    [6, 0, 4, 0, 4, 0, 6, 0, 4, 0, 4, 0, 6, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4],
], dtype=int)

T_OUT = np.array([
    [0, 4, 0, 6, 0, 4, 0, 4, 0, 6, 0, 4, 0, 4, 0, 6, 0],
    [6, 4, 4, 6, 4, 4, 6, 4, 4, 6, 4, 4, 6, 4, 4, 6, 4],
    [6, 0, 4, 0, 4, 0, 6, 0, 4, 0, 4, 0, 6, 0, 4, 0, 4],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
 for A in j:A[::3]=[6 if v==4 else v for v in A[::3]]
 return j


# --- Code Golf Solution (Compressed) ---
def q(g, v=0):
    return g * 0 != 0 and [*map(p, g, b'\n\x08\x08' * 7)] or -g % v


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

def recolor(
    value: Integer,
    patch: Patch
) -> Object:
    """ recolor patch """
    return frozenset((value, index) for index in toindices(patch))

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

def generate_ba26e723(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (0, 6))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    gi = canvas(0, (h, w))
    go = canvas(0, (h, w))
    opts = interval(0, h, 1)
    ncols = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(cols, ncols)
    for j in range(w):
        nc = unifint(diff_lb, diff_ub, (1, h - 1))
        locs = sample(opts, nc)
        obj = frozenset({(choice(ccols), (ii, j)) for ii in locs})
        gi = paint(gi, obj)
        if j % 3 == 0:
            obj = recolor(6, obj)
        go = paint(go, obj)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Numerical = Union[Integer, IntegerTuple]

ZERO = 0

THREE = 3

SIX = 6

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

def compose(
    outer: Callable,
    inner: Callable
) -> Callable:
    """ function composition """
    return lambda x: outer(inner(x))

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

def fork(
    outer: Callable,
    a: Callable,
    b: Callable
) -> Callable:
    """ creates a wrapper function """
    return lambda x: outer(a(x), b(x))

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

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_ba26e723(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = asobject(I)
    x1 = matcher(first, ZERO)
    x2 = compose(flip, x1)
    x3 = sfilter(x0, x2)
    x4 = rbind(multiply, THREE)
    x5 = rbind(divide, THREE)
    x6 = compose(x4, x5)
    x7 = fork(equality, identity, x6)
    x8 = toindices(x3)
    x9 = compose(x7, last)
    x10 = sfilter(x8, x9)
    x11 = fill(I, SIX, x10)
    return x11


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("E4", E4_IN, E4_OUT),
        ("E5", E5_IN, E5_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_ba26e723(inp)
        assert pred == _to_grid(expected), f"{name} failed"

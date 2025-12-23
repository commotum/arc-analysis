# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "017c7c7b"
SERIAL = "003"
URL    = "https://arcprize.org/play?task=017c7c7b"

# --- Code Golf Concepts ---
CONCEPTS = [
    "recoloring",
    "pattern_expansion",
    "pattern_repetition",
    "image_expansion",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 1, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 1, 1],
    [0, 1, 0],
    [1, 1, 0],
], dtype=int)

E1_OUT = np.array([
    [0, 2, 0],
    [2, 2, 0],
    [0, 2, 0],
    [0, 2, 2],
    [0, 2, 0],
    [2, 2, 0],
    [0, 2, 0],
    [0, 2, 2],
    [0, 2, 0],
], dtype=int)

E2_IN = np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1],
], dtype=int)

E2_OUT = np.array([
    [0, 2, 0],
    [2, 0, 2],
    [0, 2, 0],
    [2, 0, 2],
    [0, 2, 0],
    [2, 0, 2],
    [0, 2, 0],
    [2, 0, 2],
    [0, 2, 0],
], dtype=int)

E3_IN = np.array([
    [0, 1, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 1, 0],
], dtype=int)

E3_OUT = np.array([
    [0, 2, 0],
    [2, 2, 0],
    [0, 2, 0],
    [0, 2, 0],
    [2, 2, 0],
    [0, 2, 0],
    [0, 2, 0],
    [2, 2, 0],
    [0, 2, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 0],
], dtype=int)

T_OUT = np.array([
    [2, 2, 2],
    [0, 2, 0],
    [0, 2, 0],
    [2, 2, 2],
    [0, 2, 0],
    [0, 2, 0],
    [2, 2, 2],
    [0, 2, 0],
    [0, 2, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
    return [[c * 2 for c in r] for r in j + (j[:3], j[2:5])[j[1] != j[4]]]


# --- Code Golf Solution (Compressed) ---
def q(g):
    return [[v * 2 for v in r] for r in g + g[g[0] == g[3]:][2:5]]


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

def replace(
    grid: Grid,
    replacee: Integer,
    replacer: Integer
) -> Grid:
    """ color substitution """
    return tuple(tuple(replacer if v == replacee else v for v in r) for r in grid)

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

def generate_017c7c7b(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (0, 2))
    h = unifint(diff_lb, diff_ub, (3, 10))
    w = unifint(diff_lb, diff_ub, (2, 30))
    h += h
    fgc = choice(cols)
    go = canvas(0, (h + h // 2, w))
    oh = unifint(diff_lb, diff_ub, (1, h//3*2))
    ow = unifint(diff_lb, diff_ub, (1, w))
    locj = randint(0, w - ow)
    bounds = asindices(canvas(-1, (oh, ow)))
    ncellsd = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    ncells = choice((ncellsd, oh * ow - ncellsd))
    ncells = min(max(1, ncells), oh * ow)
    obj = sample(totuple(bounds), ncells)
    for k in range((2*h)//oh):
        go = fill(go, 2, shift(obj, (k*oh, 0)))
    gi = replace(go[:h], 2, fgc)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Numerical = Union[Integer, IntegerTuple]

IntegerSet = FrozenSet[Integer]

Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

ContainerContainer = Container[Container]

ZERO = 0

TWO = 2

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

def halve(
    n: Numerical
) -> Numerical:
    """ scaling by one half """
    return n // 2 if isinstance(n, int) else (n[0] // 2, n[1] // 2)

def merge(
    containers: ContainerContainer
) -> Container:
    """ merging """
    return type(containers)(e for c in containers for e in c)

def increment(
    x: Numerical
) -> Numerical:
    """ incrementing """
    return x + 1 if isinstance(x, int) else (x[0] + 1, x[1] + 1)

def toivec(
    i: Integer
) -> IntegerTuple:
    """ vector pointing vertically """
    return (i, 0)

def first(
    container: Container
) -> Any:
    """ first item of container """
    return next(iter(container))

def remove(
    value: Any,
    container: Container
) -> Container:
    """ remove item from container """
    return type(container)(e for e in container if e != value)

def other(
    container: Container,
    value: Any
) -> Any:
    """ other value in the container """
    return first(remove(value, container))

def astuple(
    a: Integer,
    b: Integer
) -> IntegerTuple:
    """ constructs a tuple """
    return (a, b)

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

def apply(
    function: Callable,
    container: Container
) -> Container:
    """ apply function to each item in container """
    return type(container)(function(e) for e in container)

def mapply(
    function: Callable,
    container: ContainerContainer
) -> FrozenSet:
    """ apply and merge """
    return merge(apply(function, container))

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

def normalize(
    patch: Patch
) -> Patch:
    """ moves upper left corner to origin """
    if len(patch) == 0:
        return patch
    return shift(patch, (-uppermost(patch), -leftmost(patch)))

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

def asobject(
    grid: Grid
) -> Object:
    """ conversion of grid to object """
    return frozenset((v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r))

def vperiod(
    obj: Object
) -> Integer:
    """ vertical periodicity """
    normalized = normalize(obj)
    h = height(normalized)
    for p in range(1, h):
        offsetted = shift(normalized, (-p, 0))
        pruned = frozenset({(c, (i, j)) for c, (i, j) in offsetted if i >= 0})
        if pruned.issubset(normalized):
            return p
    return h

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_017c7c7b(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = palette(I)
    x1 = other(x0, ZERO)
    x2 = ofcolor(I, x1)
    x3 = asobject(I)
    x4 = vperiod(x3)
    x5 = height(I)
    x6 = halve(x5)
    x7 = add(x5, x6)
    x8 = width(I)
    x9 = astuple(x7, x8)
    x10 = canvas(ZERO, x9)
    x11 = increment(x7)
    x12 = interval(ZERO, x11, x4)
    x13 = lbind(shift, x2)
    x14 = apply(toivec, x12)
    x15 = mapply(x13, x14)
    x16 = fill(x10, TWO, x15)
    return x16


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_017c7c7b(inp)
        assert pred == _to_grid(expected), f"{name} failed"

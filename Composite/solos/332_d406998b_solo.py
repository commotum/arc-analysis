# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "d406998b"
SERIAL = "332"
URL    = "https://arcprize.org/play?task=d406998b"

# --- Code Golf Concepts ---
CONCEPTS = [
    "recoloring",
    "one_yes_one_no",
    "cylindrical",
]

# --- Example Grids ---
E1_IN = np.array([
    [5, 0, 5, 0, 0, 5, 0, 0, 0, 5],
    [0, 5, 0, 0, 5, 0, 0, 5, 0, 0],
    [0, 0, 0, 5, 0, 0, 5, 0, 5, 0],
], dtype=int)

E1_OUT = np.array([
    [5, 0, 5, 0, 0, 3, 0, 0, 0, 3],
    [0, 3, 0, 0, 5, 0, 0, 3, 0, 0],
    [0, 0, 0, 3, 0, 0, 5, 0, 5, 0],
], dtype=int)

E2_IN = np.array([
    [0, 5, 0, 5, 0, 0, 5, 0, 5, 0, 0, 0],
    [5, 0, 0, 0, 5, 0, 0, 5, 0, 0, 5, 0],
    [0, 0, 5, 0, 0, 5, 0, 0, 0, 5, 0, 5],
], dtype=int)

E2_OUT = np.array([
    [0, 3, 0, 3, 0, 0, 5, 0, 5, 0, 0, 0],
    [5, 0, 0, 0, 5, 0, 0, 3, 0, 0, 5, 0],
    [0, 0, 5, 0, 0, 3, 0, 0, 0, 3, 0, 3],
], dtype=int)

E3_IN = np.array([
    [0, 0, 5, 0, 0, 5, 0, 5, 0, 0, 0, 5, 0],
    [5, 0, 0, 0, 5, 0, 5, 0, 0, 5, 0, 0, 5],
    [0, 5, 0, 5, 0, 0, 0, 0, 5, 0, 5, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [0, 0, 3, 0, 0, 5, 0, 5, 0, 0, 0, 5, 0],
    [3, 0, 0, 0, 3, 0, 3, 0, 0, 5, 0, 0, 3],
    [0, 5, 0, 5, 0, 0, 0, 0, 3, 0, 3, 0, 0],
], dtype=int)

E4_IN = np.array([
    [0, 0, 5, 0, 0, 5, 0, 5, 0, 5, 0, 5, 0, 0],
    [5, 0, 0, 0, 5, 0, 0, 0, 5, 0, 5, 0, 0, 5],
    [0, 5, 0, 5, 0, 0, 5, 0, 0, 0, 0, 0, 5, 0],
], dtype=int)

E4_OUT = np.array([
    [0, 0, 5, 0, 0, 3, 0, 3, 0, 3, 0, 3, 0, 0],
    [5, 0, 0, 0, 5, 0, 0, 0, 5, 0, 5, 0, 0, 3],
    [0, 3, 0, 3, 0, 0, 5, 0, 0, 0, 0, 0, 5, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 5, 0, 0, 5, 0, 5, 0, 0],
    [5, 0, 5, 0, 0, 5, 0, 0, 5, 0, 0, 5, 0, 0, 0, 5, 0],
    [0, 5, 0, 0, 5, 0, 5, 0, 0, 0, 5, 0, 0, 5, 0, 0, 5],
], dtype=int)

T_OUT = np.array([
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 5, 0, 0, 3, 0, 3, 0, 0],
    [3, 0, 3, 0, 0, 5, 0, 0, 3, 0, 0, 5, 0, 0, 0, 5, 0],
    [0, 5, 0, 0, 3, 0, 3, 0, 0, 0, 3, 0, 0, 5, 0, 0, 3],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(g):
    return [[3 if g[i][j] == 5 and (len(g[0]) - 1 - j) % 2 == 0 else g[i][j] for j in range(len(g[0]))] for i in range(3)]


# --- Code Golf Solution (Compressed) ---
def q(g):
    return [[r.pop(0) * -7 ** len(r) & 7 for x in r * 1] for r in g]


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import sample, uniform

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

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

def generate_d406998b(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(3, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    bgc, dotc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    itv = interval(0, h, 1)
    for j in range(w):
        nilocs = unifint(diff_lb, diff_ub, (1, h // 2 - 1 if h % 2 == 0 else h // 2))
        ilocs = sample(itv, nilocs)
        locs = {(ii, j) for ii in ilocs}
        gi = fill(gi, dotc, locs)
        go = fill(go, dotc if (j - w) % 2 == 0 else 3, locs)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Numerical = Union[Integer, IntegerTuple]

IntegerSet = FrozenSet[Integer]

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

ContainerContainer = Container[Container]

THREE = 3

def identity(
    x: Any
) -> Any:
    """ identity function """
    return x

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

def equality(
    a: Any,
    b: Any
) -> Boolean:
    """ equality """
    return a == b

def merge(
    containers: ContainerContainer
) -> Container:
    """ merging """
    return type(containers)(e for c in containers for e in c)

def sfilter(
    container: Container,
    condition: Callable
) -> Container:
    """ keep elements in container that satisfy condition """
    return type(container)(e for e in container if condition(e))

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

def fgpartition(
    grid: Grid
) -> Objects:
    """ each cell with the same value part of the same object without background """
    return frozenset(
        frozenset(
            (v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value
        ) for value in palette(grid) - {mostcolor(grid)}
    )

def palette(
    element: Element
) -> IntegerSet:
    """ colors occurring in object or grid """
    if isinstance(element, tuple):
        return frozenset({v for r in element for v in r})
    return frozenset({v for v, _ in element})

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

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_d406998b(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = vmirror(I)
    x1 = fgpartition(x0)
    x2 = merge(x1)
    x3 = toindices(x2)
    x4 = compose(double, halve)
    x5 = fork(equality, identity, x4)
    x6 = compose(x5, last)
    x7 = sfilter(x3, x6)
    x8 = fill(x0, THREE, x7)
    x9 = vmirror(x8)
    return x9


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("E4", E4_IN, E4_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_d406998b(inp)
        assert pred == _to_grid(expected), f"{name} failed"

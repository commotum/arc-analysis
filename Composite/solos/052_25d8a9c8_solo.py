# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "25d8a9c8"
SERIAL = "052"
URL    = "https://arcprize.org/play?task=25d8a9c8"

# --- Code Golf Concepts ---
CONCEPTS = [
    "detect_hor_lines",
    "recoloring",
    "remove_noise",
]

# --- Example Grids ---
E1_IN = np.array([
    [4, 4, 4],
    [2, 3, 2],
    [2, 3, 3],
], dtype=int)

E1_OUT = np.array([
    [5, 5, 5],
    [0, 0, 0],
    [0, 0, 0],
], dtype=int)

E2_IN = np.array([
    [7, 3, 3],
    [6, 6, 6],
    [3, 7, 7],
], dtype=int)

E2_OUT = np.array([
    [0, 0, 0],
    [5, 5, 5],
    [0, 0, 0],
], dtype=int)

E3_IN = np.array([
    [2, 9, 2],
    [4, 4, 4],
    [9, 9, 9],
], dtype=int)

E3_OUT = np.array([
    [0, 0, 0],
    [5, 5, 5],
    [5, 5, 5],
], dtype=int)

E4_IN = np.array([
    [2, 2, 4],
    [2, 2, 4],
    [1, 1, 1],
], dtype=int)

E4_OUT = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [5, 5, 5],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [4, 4, 4],
    [3, 2, 3],
    [8, 8, 8],
], dtype=int)

T_OUT = np.array([
    [5, 5, 5],
    [0, 0, 0],
    [5, 5, 5],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
    return [[5] * 3 if len(set(r)) == 1 else [0] * 3 for r in j]


# --- Code Golf Solution (Compressed) ---
def q(m):
    return [[len({*r}) % 2 * 5] * 3 for r in m]


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, sample, uniform

Integer = int

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

def generate_25d8a9c8(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    gi = []
    go = []
    ncols = unifint(diff_lb, diff_ub, (2, 10))
    ccols = sample(cols, ncols)
    for k in range(h):
        singlecol = choice((True, False))
        col = choice(ccols)
        row = repeat(col, w)
        if singlecol:
            gi.append(row)
            go.append(repeat(5, w))
        else:
            remcols = remove(col, ccols)
            nothercinv = unifint(diff_lb, diff_ub, (1, w - 1))
            notherc = w - 1 - nothercinv
            notherc = min(max(1, notherc), w - 1)
            row = list(row)
            indss = interval(0, w, 1)
            for j in sample(indss, notherc):
                row[j] = choice(remcols)
            gi.append(tuple(row))
            go.append(repeat(0, w))
    gi = tuple(gi)
    go = tuple(go)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

IntegerTuple = Tuple[Integer, Integer]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

Piece = Union[Grid, Patch]

ZERO = 0

ONE = 1

FIVE = 5

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

def toindices(
    patch: Patch
) -> Indices:
    """ indices of object cells """
    if len(patch) == 0:
        return frozenset()
    if isinstance(next(iter(patch))[1], tuple):
        return frozenset(index for value, index in patch)
    return patch

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

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_25d8a9c8(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = width(I)
    x1 = rbind(branch, ZERO)
    x2 = rbind(x1, FIVE)
    x3 = compose(size, dedupe)
    x4 = matcher(x3, ONE)
    x5 = compose(x2, x4)
    x6 = rbind(repeat, x0)
    x7 = compose(x6, x5)
    x8 = apply(x7, I)
    return x8


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("E4", E4_IN, E4_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_25d8a9c8(inp)
        assert pred == _to_grid(expected), f"{name} failed"

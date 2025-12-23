# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "c3e719e8"
SERIAL = "304"
URL    = "https://arcprize.org/play?task=c3e719e8"

# --- Code Golf Concepts ---
CONCEPTS = [
    "image_repetition",
    "image_expansion",
    "count_different_colors",
    "take_maximum",
]

# --- Example Grids ---
E1_IN = np.array([
    [3, 8, 7],
    [9, 3, 8],
    [7, 9, 3],
], dtype=int)

E1_OUT = np.array([
    [3, 8, 7, 0, 0, 0, 0, 0, 0],
    [9, 3, 8, 0, 0, 0, 0, 0, 0],
    [7, 9, 3, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 3, 8, 7, 0, 0, 0],
    [0, 0, 0, 9, 3, 8, 0, 0, 0],
    [0, 0, 0, 7, 9, 3, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 3, 8, 7],
    [0, 0, 0, 0, 0, 0, 9, 3, 8],
    [0, 0, 0, 0, 0, 0, 7, 9, 3],
], dtype=int)

E2_IN = np.array([
    [8, 6, 8],
    [3, 3, 8],
    [8, 8, 8],
], dtype=int)

E2_OUT = np.array([
    [8, 6, 8, 0, 0, 0, 8, 6, 8],
    [3, 3, 8, 0, 0, 0, 3, 3, 8],
    [8, 8, 8, 0, 0, 0, 8, 8, 8],
    [0, 0, 0, 0, 0, 0, 8, 6, 8],
    [0, 0, 0, 0, 0, 0, 3, 3, 8],
    [0, 0, 0, 0, 0, 0, 8, 8, 8],
    [8, 6, 8, 8, 6, 8, 8, 6, 8],
    [3, 3, 8, 3, 3, 8, 3, 3, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 8],
], dtype=int)

E3_IN = np.array([
    [6, 9, 9],
    [4, 6, 8],
    [9, 9, 8],
], dtype=int)

E3_OUT = np.array([
    [0, 0, 0, 6, 9, 9, 6, 9, 9],
    [0, 0, 0, 4, 6, 8, 4, 6, 8],
    [0, 0, 0, 9, 9, 8, 9, 9, 8],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [6, 9, 9, 6, 9, 9, 0, 0, 0],
    [4, 6, 8, 4, 6, 8, 0, 0, 0],
    [9, 9, 8, 9, 9, 8, 0, 0, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [1, 1, 7],
    [7, 4, 1],
    [5, 1, 7],
], dtype=int)

T_OUT = np.array([
    [1, 1, 7, 1, 1, 7, 0, 0, 0],
    [7, 4, 1, 7, 4, 1, 0, 0, 0],
    [5, 1, 7, 5, 1, 7, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 7],
    [0, 0, 0, 0, 0, 0, 7, 4, 1],
    [0, 0, 0, 0, 0, 0, 5, 1, 7],
    [0, 0, 0, 1, 1, 7, 0, 0, 0],
    [0, 0, 0, 7, 4, 1, 0, 0, 0],
    [0, 0, 0, 5, 1, 7, 0, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j,A=range(9),c=range(3)):
 E,k=__import__('collections').Counter(j[0]+j[1]+j[2]).most_common(1)[0][0],[[0 for _ in A]for _ in A]
 for W,l in[(W,l)for l in c for W in c if j[W][l]==E]:
  for J in A:k[3*W+J%3][3*l+J//3]=j[J%3][J//3]
 return k


# --- Code Golf Solution (Compressed) ---
def q(m):
    return [[a * (b == max((f := sum(m, m)), key=f.count)) for b in y for a in x] for y in m for x in m]


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, randint, sample, uniform

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Numerical = Union[Integer, IntegerTuple]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

ContainerContainer = Container[Container]

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

def difference(
    a: Container,
    b: Container
) -> Container:
    """ difference """
    return type(a)(e for e in a if e not in b)

def merge(
    containers: ContainerContainer
) -> Container:
    """ merging """
    return type(containers)(e for c in containers for e in c)

def totuple(
    container: FrozenSet
) -> Tuple:
    """ conversion to tuple """
    return tuple(container)

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

def asindices(
    grid: Grid
) -> Indices:
    """ indices of all grid cells """
    return frozenset((i, j) for i in range(len(grid)) for j in range(len(grid[0])))

def ofcolor(
    grid: Grid,
    value: Integer
) -> Indices:
    """ indices of all grid cells with value """
    return frozenset((i, j) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value)

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

def generate_c3e719e8(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(0, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 5))
    w = unifint(diff_lb, diff_ub, (2, 5))
    gob = canvas(-1, (h**2, w**2))
    wg = canvas(-1, (h, w))
    ncols = unifint(diff_lb, diff_ub, (1, min(h * w - 1, 8)))
    nmc = randint(max(1, (h * w) // (ncols + 1) + 1), h * w)
    inds = totuple(asindices(wg))
    mc = choice(cols)
    remcols = remove(mc, cols)
    mcc = sample(inds, nmc)
    inds = difference(inds, mcc)
    gi = fill(wg, mc, mcc)
    ocols = sample(remcols, ncols)
    k = len(inds) // ncols + 1
    for ocol in ocols:
        if len(inds) == 0:
            break
        ub = min(nmc - 1, len(inds))
        ub = min(ub, k)
        ub = max(ub, 1)
        locs = sample(inds, unifint(diff_lb, diff_ub, (1, ub)))
        inds = difference(inds, locs)
        gi = fill(gi, ocol, locs)
    gi = replace(gi, -1, mc)
    o = asobject(gi)
    gob = replace(gob, -1, 0)
    go = paint(gob, mapply(lbind(shift, o), apply(rbind(multiply, (h, w)), ofcolor(gi, mc))))
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

ZERO = 0

def mostcolor(
    element: Element
) -> Integer:
    """ most common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return max(set(values), key=values.count)

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

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_c3e719e8(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = shape(I)
    x1 = multiply(x0, x0)
    x2 = canvas(ZERO, x1)
    x3 = mostcolor(I)
    x4 = ofcolor(I, x3)
    x5 = lbind(multiply, x0)
    x6 = apply(x5, x4)
    x7 = asobject(I)
    x8 = lbind(shift, x7)
    x9 = mapply(x8, x6)
    x10 = paint(x2, x9)
    return x10


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_c3e719e8(inp)
        assert pred == _to_grid(expected), f"{name} failed"

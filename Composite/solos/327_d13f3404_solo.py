# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "d13f3404"
SERIAL = "327"
URL    = "https://arcprize.org/play?task=d13f3404"

# --- Code Golf Concepts ---
CONCEPTS = [
    "image_expansion",
    "draw_line_from_point",
    "diagonals",
]

# --- Example Grids ---
E1_IN = np.array([
    [6, 1, 0],
    [3, 0, 0],
    [0, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [6, 1, 0, 0, 0, 0],
    [3, 6, 1, 0, 0, 0],
    [0, 3, 6, 1, 0, 0],
    [0, 0, 3, 6, 1, 0],
    [0, 0, 0, 3, 6, 1],
    [0, 0, 0, 0, 3, 6],
], dtype=int)

E2_IN = np.array([
    [0, 4, 0],
    [0, 8, 0],
    [2, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [0, 4, 0, 0, 0, 0],
    [0, 8, 4, 0, 0, 0],
    [2, 0, 8, 4, 0, 0],
    [0, 2, 0, 8, 4, 0],
    [0, 0, 2, 0, 8, 4],
    [0, 0, 0, 2, 0, 8],
], dtype=int)

E3_IN = np.array([
    [0, 0, 6],
    [1, 3, 0],
    [0, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [0, 0, 6, 0, 0, 0],
    [1, 3, 0, 6, 0, 0],
    [0, 1, 3, 0, 6, 0],
    [0, 0, 1, 3, 0, 6],
    [0, 0, 0, 1, 3, 0],
    [0, 0, 0, 0, 1, 3],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 3],
    [0, 0, 0],
    [0, 4, 9],
], dtype=int)

T_OUT = np.array([
    [0, 0, 3, 0, 0, 0],
    [0, 0, 0, 3, 0, 0],
    [0, 4, 9, 0, 3, 0],
    [0, 0, 4, 9, 0, 3],
    [0, 0, 0, 4, 9, 0],
    [0, 0, 0, 0, 4, 9],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(g,e=enumerate):X=[[0]*6 for _ in[0]*6];[X[r+i].__setitem__(c+i,v)for r,R in e(g)for c,v in e(R)if v for i in range(6-max(r,c))];return X


# --- Code Golf Solution (Compressed) ---
def q(m, a=[0] * 3):
    return [(a := [*map(max, [0] + a * 2, r + [0] * 3)]) for r in m + [a] * 3]


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

def shoot(
    start: IntegerTuple,
    direction: IntegerTuple
) -> Indices:
    """ line from starting point and direction """
    return connect(start, (start[0] + 42 * direction[0], start[1] + 42 * direction[1]))

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

def generate_d13f3404(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 15))
    w = unifint(diff_lb, diff_ub, (3, 15))
    vopts = {(ii, 0) for ii in interval(0, h, 1)}
    hopts = {(0, jj) for jj in interval(1, w, 1)}
    opts = tuple(vopts | hopts)
    num = unifint(diff_lb, diff_ub, (1, len(opts)))
    locs = sample(opts, num)
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h*2, w*2))
    inds = asindices(gi)
    for loc in locs:
        ln = tuple(shoot(loc, (1, 1)) & inds)
        locc = choice(ln)
        col = choice(remcols)
        gi = fill(gi, col, {locc})
        go = fill(go, col, shoot(locc, (1, 1)))
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Numerical = Union[Integer, IntegerTuple]

Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

ContainerContainer = Container[Container]

UNITY = (1, 1)

def double(
    n: Numerical
) -> Numerical:
    """ scaling by two """
    return n * 2 if isinstance(n, int) else (n[0] * 2, n[1] * 2)

def flip(
    b: Boolean
) -> Boolean:
    """ logical not """
    return not b

def merge(
    containers: ContainerContainer
) -> Container:
    """ merging """
    return type(containers)(e for c in containers for e in c)

def initset(
    value: Any
) -> FrozenSet:
    """ initialize container """
    return frozenset({value})

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

def recolor(
    value: Integer,
    patch: Patch
) -> Object:
    """ recolor patch """
    return frozenset((value, index) for index in toindices(patch))

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

def color(
    obj: Object
) -> Integer:
    """ color of object """
    return next(iter(obj))[0]

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

def center(
    patch: Patch
) -> IntegerTuple:
    """ center of the patch """
    return (uppermost(patch) + height(patch) // 2, leftmost(patch) + width(patch) // 2)

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_d13f3404(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = asobject(I)
    x1 = mostcolor(I)
    x2 = matcher(first, x1)
    x3 = compose(flip, x2)
    x4 = sfilter(x0, x3)
    x5 = apply(initset, x4)
    x6 = rbind(shoot, UNITY)
    x7 = compose(x6, center)
    x8 = fork(recolor, color, x7)
    x9 = mapply(x8, x5)
    x10 = shape(I)
    x11 = double(x10)
    x12 = mostcolor(I)
    x13 = canvas(x12, x11)
    x14 = paint(x13, x9)
    return x14


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_d13f3404(inp)
        assert pred == _to_grid(expected), f"{name} failed"

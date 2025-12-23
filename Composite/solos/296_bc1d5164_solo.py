# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "bc1d5164"
SERIAL = "296"
URL    = "https://arcprize.org/play?task=bc1d5164"

# --- Code Golf Concepts ---
CONCEPTS = [
    "pattern_moving",
    "pattern_juxtaposition",
    "crop",
    "pairwise_analogy",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 8, 0, 0, 0, 8, 0],
    [8, 8, 0, 0, 0, 8, 8],
    [0, 0, 0, 0, 0, 0, 0],
    [8, 8, 0, 0, 0, 8, 8],
    [0, 8, 0, 0, 0, 8, 0],
], dtype=int)

E1_OUT = np.array([
    [0, 8, 0],
    [8, 8, 8],
    [0, 8, 0],
], dtype=int)

E2_IN = np.array([
    [2, 2, 0, 0, 0, 2, 2],
    [0, 0, 0, 0, 0, 0, 2],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 2, 0],
    [2, 0, 0, 0, 0, 0, 2],
], dtype=int)

E2_OUT = np.array([
    [2, 2, 2],
    [0, 2, 2],
    [2, 0, 2],
], dtype=int)

E3_IN = np.array([
    [4, 4, 0, 0, 0, 4, 0],
    [0, 0, 0, 0, 0, 4, 4],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [4, 0, 0, 0, 0, 0, 4],
], dtype=int)

E3_OUT = np.array([
    [4, 4, 0],
    [0, 4, 4],
    [4, 0, 4],
], dtype=int)

E4_IN = np.array([
    [4, 0, 0, 0, 0, 0, 4],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [4, 0, 0, 0, 0, 4, 4],
], dtype=int)

E4_OUT = np.array([
    [4, 0, 4],
    [0, 0, 0],
    [4, 4, 4],
], dtype=int)

E5_IN = np.array([
    [0, 3, 0, 0, 0, 3, 0],
    [3, 0, 0, 0, 0, 0, 3],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 3],
], dtype=int)

E5_OUT = np.array([
    [0, 3, 0],
    [3, 0, 3],
    [0, 0, 3],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 0, 0, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 1],
], dtype=int)

T_OUT = np.array([
    [0, 1, 1],
    [1, 0, 0],
    [0, 1, 1],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
 A=[[0]*3,[0]*3,[0]*3]
 for c in range(16):E,k=c//8%2*-2+c//2%2,c//4%2*-2+c%2;A[E][k]=max(A[E][k],j[E][k])
 return A


# --- Code Golf Solution (Compressed) ---
def q(*g):
    return [*map([*g, max, p][2], *[r[-3:] for r in g], *g)]


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import randint, sample, uniform

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

def sfilter(
    container: Container,
    condition: Callable
) -> Container:
    """ keep elements in container that satisfy condition """
    return type(container)(e for e in container if condition(e))

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

def generate_bc1d5164(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 15))
    w = unifint(diff_lb, diff_ub, (2, 14))
    fullh = 2 * h - 1
    fullw = 2 * w + 1
    bgc, objc = sample(cols, 2)
    inds = asindices(canvas(-1, (h, w)))
    nA = randint(1, (h - 1) * (w - 1) - 1)
    nB = randint(1, (h - 1) * (w - 1) - 1)
    nC = randint(1, (h - 1) * (w - 1) - 1)
    nD = randint(1, (h - 1) * (w - 1) - 1)
    A = sample(totuple(sfilter(inds, lambda ij: ij[0] < h - 1 and ij[1] < w - 1)), nA)
    B = sample(totuple(sfilter(inds, lambda ij: ij[0] < h - 1 and ij[1] > 0)), nB)
    C = sample(totuple(sfilter(inds, lambda ij: ij[0] > 0 and ij[1] < w - 1)), nC)
    D = sample(totuple(sfilter(inds, lambda ij: ij[0] > 0 and ij[1] > 0)), nD)
    gi = canvas(bgc, (fullh, fullw))
    gi = fill(gi, objc, A)
    gi = fill(gi, objc, shift(B, (0, fullw - w)))
    gi = fill(gi, objc, shift(C, (fullh - h, 0)))
    gi = fill(gi, objc, shift(D, (fullh - h, fullw - w)))
    go = canvas(bgc, (h, w))
    go = fill(go, objc, set(A) | set(B) | set(C) | set(D))
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Numerical = Union[Integer, IntegerTuple]

IntegerSet = FrozenSet[Integer]

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

ContainerContainer = Container[Container]

def halve(
    n: Numerical
) -> Numerical:
    """ scaling by one half """
    return n // 2 if isinstance(n, int) else (n[0] // 2, n[1] // 2)

def combine(
    a: Container,
    b: Container
) -> Container:
    """ union """
    return type(a)((*a, *b))

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

def decrement(
    x: Numerical
) -> Numerical:
    """ decrementing """
    return x - 1 if isinstance(x, int) else (x[0] - 1, x[1] - 1)

def toivec(
    i: Integer
) -> IntegerTuple:
    """ vector pointing vertically """
    return (i, 0)

def tojvec(
    j: Integer
) -> IntegerTuple:
    """ vector pointing horizontally """
    return (0, j)

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

def ulcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of upper left corner """
    return tuple(map(min, zip(*toindices(patch))))

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

def toobject(
    patch: Patch,
    grid: Grid
) -> Object:
    """ object from patch and grid """
    h, w = len(grid), len(grid[0])
    return frozenset((grid[i][j], (i, j)) for i, j in toindices(patch) if 0 <= i < h and 0 <= j < w)

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

def frontiers(
    grid: Grid
) -> Objects:
    """ set of frontiers """
    h, w = len(grid), len(grid[0])
    row_indices = tuple(i for i, r in enumerate(grid) if len(set(r)) == 1)
    column_indices = tuple(j for j, c in enumerate(dmirror(grid)) if len(set(c)) == 1)
    hfrontiers = frozenset({frozenset({(grid[i][j], (i, j)) for j in range(w)}) for i in row_indices})
    vfrontiers = frozenset({frozenset({(grid[i][j], (i, j)) for i in range(h)}) for j in column_indices})
    return hfrontiers | vfrontiers

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_bc1d5164(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = height(I)
    x1 = halve(x0)
    x2 = increment(x1)
    x3 = width(I)
    x4 = halve(x3)
    x5 = frontiers(I)
    x6 = merge(x5)
    x7 = mostcolor(x6)
    x8 = astuple(x2, x4)
    x9 = canvas(x7, x8)
    x10 = asindices(x9)
    x11 = toobject(x10, I)
    x12 = increment(x4)
    x13 = tojvec(x12)
    x14 = shift(x10, x13)
    x15 = toobject(x14, I)
    x16 = decrement(x2)
    x17 = toivec(x16)
    x18 = shift(x10, x17)
    x19 = toobject(x18, I)
    x20 = decrement(x2)
    x21 = increment(x4)
    x22 = astuple(x20, x21)
    x23 = shift(x10, x22)
    x24 = toobject(x23, I)
    x25 = palette(I)
    x26 = other(x25, x7)
    x27 = matcher(first, x26)
    x28 = rbind(sfilter, x27)
    x29 = chain(toindices, x28, normalize)
    x30 = x29(x11)
    x31 = x29(x15)
    x32 = x29(x19)
    x33 = x29(x24)
    x34 = combine(x30, x31)
    x35 = combine(x32, x33)
    x36 = combine(x34, x35)
    x37 = fill(x9, x26, x36)
    return x37


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
        pred = verify_bc1d5164(inp)
        assert pred == _to_grid(expected), f"{name} failed"

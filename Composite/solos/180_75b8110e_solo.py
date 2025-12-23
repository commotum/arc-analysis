# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "75b8110e"
SERIAL = "180"
URL    = "https://arcprize.org/play?task=75b8110e"

# --- Code Golf Concepts ---
CONCEPTS = [
    "separate_images",
    "image_juxtaposition",
]

# --- Example Grids ---
E1_IN = np.array([
    [4, 4, 0, 0, 0, 0, 5, 0],
    [4, 4, 0, 0, 0, 0, 0, 0],
    [0, 0, 4, 0, 0, 0, 5, 0],
    [0, 4, 0, 0, 5, 5, 0, 0],
    [0, 0, 6, 0, 0, 0, 9, 0],
    [6, 6, 6, 0, 0, 0, 0, 9],
    [6, 0, 6, 6, 9, 9, 0, 0],
    [0, 6, 6, 0, 9, 0, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [4, 4, 5, 0],
    [6, 6, 6, 9],
    [6, 9, 5, 6],
    [5, 5, 6, 0],
], dtype=int)

E2_IN = np.array([
    [4, 0, 0, 4, 5, 5, 0, 0],
    [0, 0, 0, 0, 0, 0, 5, 5],
    [4, 4, 0, 4, 0, 5, 0, 0],
    [4, 0, 4, 4, 0, 5, 5, 5],
    [0, 0, 0, 6, 0, 9, 0, 9],
    [0, 0, 6, 0, 0, 9, 0, 0],
    [6, 0, 0, 6, 0, 9, 0, 9],
    [0, 0, 6, 6, 0, 0, 0, 9],
], dtype=int)

E2_OUT = np.array([
    [5, 5, 0, 6],
    [0, 9, 5, 5],
    [6, 5, 0, 6],
    [4, 5, 5, 5],
], dtype=int)

E3_IN = np.array([
    [0, 0, 0, 4, 5, 0, 0, 0],
    [4, 0, 0, 0, 0, 5, 0, 0],
    [0, 0, 0, 4, 0, 0, 5, 0],
    [0, 4, 0, 4, 0, 0, 5, 0],
    [6, 0, 0, 0, 0, 9, 9, 0],
    [6, 0, 0, 0, 0, 9, 0, 9],
    [6, 0, 6, 0, 9, 9, 9, 0],
    [6, 0, 6, 0, 0, 0, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [5, 9, 9, 4],
    [6, 5, 0, 9],
    [6, 9, 5, 4],
    [6, 4, 5, 4],
], dtype=int)

E4_IN = np.array([
    [4, 0, 0, 4, 0, 5, 0, 5],
    [0, 0, 4, 0, 5, 0, 0, 5],
    [0, 0, 4, 4, 0, 0, 5, 5],
    [4, 0, 0, 0, 5, 0, 0, 5],
    [6, 6, 6, 0, 9, 0, 9, 9],
    [6, 6, 6, 0, 0, 9, 9, 9],
    [6, 0, 0, 6, 9, 9, 0, 9],
    [6, 6, 0, 6, 9, 0, 9, 9],
], dtype=int)

E4_OUT = np.array([
    [6, 5, 6, 5],
    [5, 6, 6, 5],
    [6, 9, 5, 5],
    [5, 6, 9, 5],
], dtype=int)

E5_IN = np.array([
    [0, 4, 4, 4, 0, 5, 5, 5],
    [0, 0, 4, 0, 5, 5, 0, 5],
    [0, 0, 0, 0, 5, 0, 0, 0],
    [4, 0, 0, 0, 5, 0, 0, 0],
    [6, 6, 0, 6, 0, 0, 9, 9],
    [0, 0, 0, 6, 9, 0, 9, 0],
    [0, 0, 0, 6, 9, 0, 9, 9],
    [6, 6, 0, 6, 0, 9, 0, 9],
], dtype=int)

E5_OUT = np.array([
    [6, 5, 5, 5],
    [5, 5, 9, 5],
    [5, 0, 9, 6],
    [5, 6, 0, 6],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 4, 0, 4, 5, 0, 0, 0],
    [0, 4, 4, 4, 5, 0, 5, 5],
    [4, 4, 4, 0, 0, 5, 5, 5],
    [0, 0, 0, 0, 5, 0, 0, 0],
    [6, 0, 6, 6, 9, 9, 9, 0],
    [0, 0, 0, 6, 0, 9, 0, 0],
    [0, 6, 0, 0, 0, 0, 9, 9],
    [6, 0, 0, 0, 0, 9, 0, 0],
], dtype=int)

T_OUT = np.array([
    [5, 9, 6, 6],
    [5, 9, 5, 5],
    [4, 5, 5, 5],
    [5, 9, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j, A=range(4)):
    return [[j[x][y + 4] or j[x + 4][y] or j[x + 4][y + 4] or j[x][y] for y in A] for x in A]


# --- Code Golf Solution (Compressed) ---
def q(a):
    return [p(b) for *b, in map(zip, a, a[4:])] or max(sum(a + a, ())[1:], key=bool)


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

def hconcat(
    a: Grid,
    b: Grid
) -> Grid:
    """ concatenate two grids horizontally """
    return tuple(i + j for i, j in zip(a, b))

def vconcat(
    a: Grid,
    b: Grid
) -> Grid:
    """ concatenate two grids vertically """
    return a + b

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

def generate_75b8110e(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 15))
    w = unifint(diff_lb, diff_ub, (2, 15))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    c1, c2, c3, c4 = sample(remcols, 4)
    canv = canvas(bgc, (h, w))
    cels = totuple(asindices(canv))
    mp = (h * w) // 2
    nums = []
    for k in range(4):
        dev = unifint(diff_lb, diff_ub, (0, mp))
        if choice((True, False)):
            num = h * w - dev
        else:
            num = dev
        num = min(max(0, num), h * w - 1)
        nums.append(num)
    s1, s2, s3, s4 = [sample(cels, num) for num in nums]
    gi1 = fill(canv, c1, s1)
    gi2 = fill(canv, c2, s2)
    gi3 = fill(canv, c3, s3)
    gi4 = fill(canv, c4, s4)
    gi = vconcat(hconcat(gi1, gi2), hconcat(gi3, gi4))
    go = fill(gi1, c4, s4)
    go = fill(go, c3, s3)
    go = fill(go, c2, s2)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

IntegerSet = FrozenSet[Integer]

Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

def flip(
    b: Boolean
) -> Boolean:
    """ logical not """
    return not b

def intersection(
    a: FrozenSet,
    b: FrozenSet
) -> FrozenSet:
    """ returns the intersection of two containers """
    return a & b

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

def rot90(
    grid: Grid
) -> Grid:
    """ quarter clockwise rotation """
    return tuple(row for row in zip(*grid[::-1]))

def rot270(
    grid: Grid
) -> Grid:
    """ quarter anticlockwise rotation """
    return tuple(tuple(row[::-1]) for row in zip(*grid[::-1]))[::-1]

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

def tophalf(
    grid: Grid
) -> Grid:
    """ upper half of grid """
    return grid[:len(grid) // 2]

def bottomhalf(
    grid: Grid
) -> Grid:
    """ lower half of grid """
    return grid[len(grid) // 2 + len(grid) % 2:]

def lefthalf(
    grid: Grid
) -> Grid:
    """ left half of grid """
    return rot270(tophalf(rot90(grid)))

def righthalf(
    grid: Grid
) -> Grid:
    """ right half of grid """
    return rot270(bottomhalf(rot90(grid)))

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_75b8110e(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = tophalf(I)
    x1 = lefthalf(x0)
    x2 = tophalf(I)
    x3 = righthalf(x2)
    x4 = bottomhalf(I)
    x5 = righthalf(x4)
    x6 = bottomhalf(I)
    x7 = lefthalf(x6)
    x8 = palette(x1)
    x9 = palette(x3)
    x10 = intersection(x8, x9)
    x11 = palette(x5)
    x12 = palette(x7)
    x13 = intersection(x11, x12)
    x14 = intersection(x10, x13)
    x15 = first(x14)
    x16 = shape(x1)
    x17 = canvas(x15, x16)
    x18 = matcher(first, x15)
    x19 = compose(flip, x18)
    x20 = rbind(sfilter, x19)
    x21 = compose(x20, asobject)
    x22 = x21(x1)
    x23 = x21(x5)
    x24 = x21(x7)
    x25 = x21(x3)
    x26 = paint(x17, x22)
    x27 = paint(x26, x23)
    x28 = paint(x27, x24)
    x29 = paint(x28, x25)
    return x29


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
        pred = verify_75b8110e(inp)
        assert pred == _to_grid(expected), f"{name} failed"

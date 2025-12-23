# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "c9f8e694"
SERIAL = "312"
URL    = "https://arcprize.org/play?task=c9f8e694"

# --- Code Golf Concepts ---
CONCEPTS = [
    "recoloring",
    "pattern_repetition",
    "color_palette",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 0, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 5, 5, 0, 0, 0, 0, 5, 5, 0, 0],
    [2, 0, 5, 5, 0, 0, 0, 0, 5, 5, 0, 0],
    [2, 0, 5, 5, 0, 0, 0, 0, 5, 5, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 5, 5, 0, 0],
    [1, 0, 0, 0, 5, 5, 5, 0, 5, 5, 0, 0],
    [1, 0, 0, 0, 5, 5, 5, 0, 5, 5, 0, 0],
    [2, 0, 0, 0, 5, 5, 5, 0, 5, 5, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
    [2, 0, 2, 2, 0, 0, 0, 0, 2, 2, 0, 0],
    [2, 0, 2, 2, 0, 0, 0, 0, 2, 2, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    [1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0],
    [1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0],
    [2, 0, 0, 0, 2, 2, 2, 0, 2, 2, 0, 0],
], dtype=int)

E2_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0],
    [3, 5, 5, 5, 5, 0, 0, 5, 5, 5, 5, 5],
    [4, 5, 5, 5, 5, 0, 0, 5, 5, 5, 5, 5],
    [4, 5, 5, 5, 5, 0, 0, 5, 5, 5, 5, 5],
    [3, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5],
    [4, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5],
    [3, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5],
    [3, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5],
    [3, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5],
    [4, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0],
    [4, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0],
    [3, 3, 3, 3, 3, 0, 0, 3, 3, 3, 3, 3],
    [4, 4, 4, 4, 4, 0, 0, 4, 4, 4, 4, 4],
    [4, 4, 4, 4, 4, 0, 0, 4, 4, 4, 4, 4],
    [3, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3],
    [4, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4],
    [3, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3],
    [4, 0, 0, 0, 4, 4, 4, 0, 0, 0, 0, 0],
    [4, 0, 0, 0, 4, 4, 4, 0, 0, 0, 0, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [1, 0, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0],
    [8, 0, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0],
    [1, 0, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0],
    [1, 0, 5, 5, 5, 5, 5, 5, 0, 5, 5, 5],
    [7, 0, 5, 5, 5, 5, 5, 5, 0, 5, 5, 5],
    [7, 0, 5, 5, 5, 5, 5, 5, 0, 5, 5, 5],
    [7, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5],
    [7, 0, 0, 0, 0, 0, 5, 5, 5, 0, 0, 0],
    [8, 0, 0, 0, 0, 0, 5, 5, 5, 0, 0, 0],
    [8, 0, 5, 5, 5, 0, 5, 5, 5, 0, 0, 0],
    [8, 0, 5, 5, 5, 0, 5, 5, 5, 0, 0, 0],
    [8, 0, 5, 5, 5, 0, 5, 5, 5, 0, 0, 0],
], dtype=int)

T_OUT = np.array([
    [1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [8, 0, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0],
    [1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    [7, 0, 7, 7, 7, 7, 7, 7, 0, 7, 7, 7],
    [7, 0, 7, 7, 7, 7, 7, 7, 0, 7, 7, 7],
    [7, 0, 0, 0, 0, 0, 7, 7, 7, 7, 7, 7],
    [7, 0, 0, 0, 0, 0, 7, 7, 7, 0, 0, 0],
    [8, 0, 0, 0, 0, 0, 8, 8, 8, 0, 0, 0],
    [8, 0, 8, 8, 8, 0, 8, 8, 8, 0, 0, 0],
    [8, 0, 8, 8, 8, 0, 8, 8, 8, 0, 0, 0],
    [8, 0, 8, 8, 8, 0, 8, 8, 8, 0, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
 for A in j:
  for c in A:
   if c and c-5:A[:]=[c*(x==5)+x*(x!=5)for x in A];break
 return j


# --- Code Golf Solution (Compressed) ---
def q(m):
    return [[v % ~v & r[0] for v in r] for r in m]


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, randint, uniform

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

def identity(
    x: Any
) -> Any:
    """ identity function """
    return x

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

def interval(
    start: Integer,
    stop: Integer,
    step: Integer
) -> Tuple:
    """ range """
    return tuple(range(start, stop, step))

def apply(
    function: Callable,
    container: Container
) -> Container:
    """ apply function to each item in container """
    return type(container)(function(e) for e in container)

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

def toindices(
    patch: Patch
) -> Indices:
    """ indices of object cells """
    if len(patch) == 0:
        return frozenset()
    if isinstance(next(iter(patch))[1], tuple):
        return frozenset(index for value, index in patch)
    return patch

def toobject(
    patch: Patch,
    grid: Grid
) -> Object:
    """ object from patch and grid """
    h, w = len(grid), len(grid[0])
    return frozenset((grid[i][j], (i, j)) for i, j in toindices(patch) if 0 <= i < h and 0 <= j < w)

def rot90(
    grid: Grid
) -> Grid:
    """ quarter clockwise rotation """
    return tuple(row for row in zip(*grid[::-1]))

def rot180(
    grid: Grid
) -> Grid:
    """ half rotation """
    return tuple(tuple(row[::-1]) for row in grid[::-1])

def rot270(
    grid: Grid
) -> Grid:
    """ quarter anticlockwise rotation """
    return tuple(tuple(row[::-1]) for row in zip(*grid[::-1]))[::-1]

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

def hupscale(
    grid: Grid,
    factor: Integer
) -> Grid:
    """ upscale grid horizontally """
    upscaled_grid = tuple()
    for row in grid:
        upscaled_row = tuple()
        for value in row:
            upscaled_row = upscaled_row + tuple(value for num in range(factor))
        upscaled_grid = upscaled_grid + (upscaled_row,)
    return upscaled_grid

def hconcat(
    a: Grid,
    b: Grid
) -> Grid:
    """ concatenate two grids horizontally """
    return tuple(i + j for i, j in zip(a, b))

def canvas(
    value: Integer,
    dimensions: IntegerTuple
) -> Grid:
    """ grid construction """
    return tuple(tuple(value for j in range(dimensions[1])) for i in range(dimensions[0]))

def backdrop(
    patch: Patch
) -> Indices:
    """ indices in bounding box of patch """
    if len(patch) == 0:
        return frozenset({})
    indices = toindices(patch)
    si, sj = ulcorner(indices)
    ei, ej = lrcorner(patch)
    return frozenset((i, j) for i in range(si, ei + 1) for j in range(sj, ej + 1))

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

def generate_c9f8e694(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(1, 10, 1)
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    bgc = 0
    remcols = remove(bgc, cols)
    sqc = choice(remcols)
    remcols = remove(sqc, remcols)
    ncols = unifint(diff_lb, diff_ub, (1, min(h, 8)))
    nsq = unifint(diff_lb, diff_ub, (1, 8))
    gir = canvas(bgc, (h, w - 1))
    gil = tuple((choice(remcols),) for j in range(h))
    inds = asindices(gir)
    succ = 0
    fails = 0
    maxfails = nsq * 5
    while succ < nsq and fails < maxfails:
        loci = randint(0, h - 3)
        locj = randint(0, w - 3)
        lock = randint(loci+1, min(loci + max(1, 2*h//3), h - 1))
        locl = randint(locj+1, min(locj + max(1, 2*w//3), w - 1))
        bd = backdrop(frozenset({(loci, locj), (lock, locl)}))
        if bd.issubset(inds):
            gir = fill(gir, sqc, bd)
            succ += 1
            indss = inds - bd
        else:
            fails += 1
    locs = ofcolor(gir, sqc)
    gil = tuple(e if idx in apply(first, locs) else (bgc,) for idx, e in enumerate(gil))
    fullobj = toobject(locs, hupscale(gil, w))
    gi = hconcat(gil, gir)
    giro = paint(gir, fullobj)
    go = hconcat(gil, giro)
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Piece = Union[Grid, Patch]

ZERO = 0

ONE = 1

ORIGIN = (0, 0)

def combine(
    a: Container,
    b: Container
) -> Container:
    """ union """
    return type(a)((*a, *b))

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

def argmax(
    container: Container,
    compfunc: Callable
) -> Any:
    """ largest item by custom order """
    return max(container, key=compfunc, default=None)

def initset(
    value: Any
) -> FrozenSet:
    """ initialize container """
    return frozenset({value})

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

def rapply(
    functions: Container,
    value: Any
) -> Container:
    """ apply each function in container to value """
    return type(functions)(function(value) for function in functions)

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

def cmirror(
    piece: Piece
) -> Piece:
    """ mirroring along counterdiagonal """
    if isinstance(piece, tuple):
        return tuple(zip(*(r[::-1] for r in piece[::-1])))
    return vmirror(dmirror(vmirror(piece)))

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_c9f8e694(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = astuple(identity, dmirror)
    x1 = astuple(cmirror, vmirror)
    x2 = combine(x0, x1)
    x3 = compose(first, dmirror)
    x4 = chain(size, dedupe, x3)
    x5 = rbind(rapply, I)
    x6 = compose(first, x5)
    x7 = chain(x4, x6, initset)
    x8 = argmax(x2, x7)
    x9 = x8(I)
    x10 = height(x9)
    x11 = width(x9)
    x12 = ofcolor(x9, ZERO)
    x13 = astuple(x10, ONE)
    x14 = crop(x9, ORIGIN, x13)
    x15 = hupscale(x14, x11)
    x16 = fill(x15, ZERO, x12)
    x17 = x8(x16)
    return x17


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_c9f8e694(inp)
        assert pred == _to_grid(expected), f"{name} failed"

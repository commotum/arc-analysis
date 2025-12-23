# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "9ecd008a"
SERIAL = "242"
URL    = "https://arcprize.org/play?task=9ecd008a"

# --- Code Golf Concepts ---
CONCEPTS = [
    "image_filling",
    "pattern_expansion",
    "pattern_reflection",
    "pattern_rotation",
    "crop",
]

# --- Example Grids ---
E1_IN = np.array([
    [2, 1, 3, 5, 1, 1, 1, 8, 8, 1, 1, 1, 5, 3, 1, 2],
    [1, 2, 5, 7, 1, 7, 8, 8, 8, 8, 7, 1, 7, 5, 2, 1],
    [3, 5, 4, 4, 1, 8, 2, 9, 9, 2, 8, 1, 4, 4, 5, 3],
    [5, 7, 4, 4, 8, 8, 9, 2, 2, 9, 8, 8, 4, 4, 7, 5],
    [1, 1, 1, 8, 4, 4, 1, 1, 1, 1, 4, 4, 8, 1, 1, 1],
    [1, 7, 8, 8, 0, 0, 0, 9, 9, 1, 7, 4, 8, 8, 7, 1],
    [1, 8, 2, 9, 0, 0, 0, 3, 3, 1, 1, 1, 9, 2, 8, 1],
    [8, 8, 9, 2, 0, 0, 0, 1, 1, 3, 9, 1, 2, 9, 8, 8],
    [8, 8, 9, 2, 1, 9, 3, 1, 1, 3, 9, 1, 2, 9, 8, 8],
    [1, 8, 2, 9, 1, 1, 1, 3, 3, 1, 1, 1, 9, 2, 8, 1],
    [1, 7, 8, 8, 4, 7, 1, 9, 9, 1, 7, 4, 8, 8, 7, 1],
    [1, 1, 1, 8, 4, 4, 1, 1, 1, 1, 4, 4, 8, 1, 1, 1],
    [5, 7, 4, 4, 8, 8, 9, 2, 2, 9, 8, 8, 4, 4, 7, 5],
    [3, 5, 4, 4, 1, 8, 2, 9, 9, 2, 8, 1, 4, 4, 5, 3],
    [1, 2, 5, 7, 1, 7, 8, 8, 8, 8, 7, 1, 7, 5, 2, 1],
    [2, 1, 3, 5, 1, 1, 1, 8, 8, 1, 1, 1, 5, 3, 1, 2],
], dtype=int)

E1_OUT = np.array([
    [4, 7, 1],
    [1, 1, 1],
    [1, 9, 3],
], dtype=int)

E2_IN = np.array([
    [3, 3, 3, 1, 7, 7, 6, 6, 6, 6, 7, 7, 1, 3, 3, 3],
    [3, 3, 1, 3, 7, 7, 6, 1, 1, 6, 7, 7, 3, 1, 3, 3],
    [3, 1, 8, 8, 6, 6, 9, 7, 7, 9, 6, 6, 8, 8, 1, 3],
    [1, 3, 8, 5, 6, 1, 7, 9, 9, 7, 1, 6, 5, 8, 3, 1],
    [7, 7, 6, 6, 3, 3, 5, 1, 1, 5, 3, 3, 6, 6, 7, 7],
    [7, 7, 6, 1, 3, 3, 1, 1, 1, 1, 3, 3, 1, 6, 7, 7],
    [6, 6, 9, 7, 5, 1, 6, 1, 1, 6, 1, 5, 7, 9, 6, 6],
    [6, 1, 7, 9, 1, 1, 1, 4, 4, 1, 1, 1, 9, 7, 1, 6],
    [6, 1, 7, 9, 0, 0, 0, 4, 4, 1, 1, 1, 9, 7, 1, 6],
    [6, 6, 9, 7, 0, 0, 0, 1, 1, 6, 1, 5, 7, 9, 6, 6],
    [7, 7, 6, 1, 0, 0, 0, 1, 1, 1, 3, 3, 1, 6, 7, 7],
    [7, 7, 6, 6, 3, 3, 5, 1, 1, 5, 3, 3, 6, 6, 7, 7],
    [1, 3, 8, 5, 6, 1, 7, 9, 9, 7, 1, 6, 5, 8, 3, 1],
    [3, 1, 8, 8, 6, 6, 9, 7, 7, 9, 6, 6, 8, 8, 1, 3],
    [3, 3, 1, 3, 7, 7, 6, 1, 1, 6, 7, 7, 3, 1, 3, 3],
    [3, 3, 3, 1, 7, 7, 6, 6, 6, 6, 7, 7, 1, 3, 3, 3],
], dtype=int)

E2_OUT = np.array([
    [1, 1, 1],
    [5, 1, 6],
    [3, 3, 1],
], dtype=int)

E3_IN = np.array([
    [9, 3, 5, 3, 3, 9, 5, 5, 5, 5, 9, 3, 3, 5, 3, 9],
    [3, 9, 3, 6, 9, 5, 5, 8, 8, 5, 5, 9, 6, 3, 9, 3],
    [5, 3, 3, 3, 5, 5, 6, 6, 6, 6, 5, 5, 3, 3, 3, 5],
    [3, 6, 3, 6, 5, 8, 6, 6, 6, 6, 8, 5, 6, 3, 6, 3],
    [3, 9, 5, 5, 5, 5, 2, 1, 1, 2, 5, 5, 5, 5, 9, 3],
    [9, 5, 5, 8, 5, 8, 1, 6, 6, 1, 8, 5, 8, 5, 5, 9],
    [5, 5, 6, 6, 2, 1, 9, 3, 3, 9, 1, 2, 6, 6, 5, 5],
    [5, 8, 6, 6, 1, 6, 3, 9, 9, 3, 0, 0, 0, 6, 8, 5],
    [5, 8, 6, 6, 1, 6, 3, 9, 9, 3, 0, 0, 0, 6, 8, 5],
    [5, 5, 6, 6, 2, 1, 9, 3, 3, 9, 0, 0, 0, 6, 5, 5],
    [9, 5, 5, 8, 5, 8, 1, 6, 6, 1, 8, 5, 8, 5, 5, 9],
    [3, 9, 5, 5, 5, 5, 2, 1, 1, 2, 5, 5, 5, 5, 9, 3],
    [3, 6, 3, 6, 5, 8, 6, 6, 6, 6, 8, 5, 6, 3, 6, 3],
    [5, 3, 3, 3, 5, 5, 6, 6, 6, 6, 5, 5, 3, 3, 3, 5],
    [3, 9, 3, 6, 9, 5, 5, 8, 8, 5, 5, 9, 6, 3, 9, 3],
    [9, 3, 5, 3, 3, 9, 5, 5, 5, 5, 9, 3, 3, 5, 3, 9],
], dtype=int)

E3_OUT = np.array([
    [6, 1, 6],
    [6, 1, 6],
    [1, 2, 6],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [4, 8, 9, 9, 6, 6, 5, 1, 1, 5, 6, 6, 9, 9, 8, 4],
    [8, 6, 9, 9, 6, 7, 1, 5, 5, 1, 7, 6, 9, 9, 6, 8],
    [9, 9, 5, 2, 5, 1, 5, 5, 5, 5, 1, 5, 2, 5, 9, 9],
    [9, 9, 2, 2, 1, 5, 5, 9, 9, 5, 5, 1, 2, 2, 9, 9],
    [6, 6, 5, 1, 1, 4, 5, 2, 2, 5, 4, 1, 1, 5, 6, 6],
    [6, 0, 0, 0, 4, 4, 2, 7, 7, 2, 4, 4, 5, 1, 7, 6],
    [5, 0, 0, 0, 5, 2, 9, 5, 5, 9, 2, 5, 5, 5, 1, 5],
    [1, 0, 0, 0, 2, 7, 5, 9, 9, 5, 7, 2, 9, 5, 5, 1],
    [1, 5, 5, 9, 2, 7, 5, 9, 9, 5, 7, 2, 9, 5, 5, 1],
    [5, 1, 5, 5, 5, 2, 9, 5, 5, 9, 2, 5, 5, 5, 1, 5],
    [6, 7, 1, 5, 4, 4, 2, 7, 7, 2, 4, 4, 5, 1, 7, 6],
    [6, 6, 5, 1, 1, 4, 5, 2, 2, 5, 4, 1, 1, 5, 6, 6],
    [9, 9, 2, 2, 1, 5, 5, 9, 9, 5, 5, 1, 2, 2, 9, 9],
    [9, 9, 5, 2, 5, 1, 5, 5, 5, 5, 1, 5, 2, 5, 9, 9],
    [8, 6, 9, 9, 6, 7, 1, 5, 5, 1, 7, 6, 9, 9, 6, 8],
    [4, 8, 9, 9, 6, 6, 5, 1, 1, 5, 6, 6, 9, 9, 8, 4],
], dtype=int)

T_OUT = np.array([
    [7, 1, 5],
    [1, 5, 5],
    [5, 5, 9],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(g,L=len,R=range):
 h,w,I,J=L(g),L(g[0]),[],[]
 for r in R(h//2+1):
  for c in R(w):
   if g[r][c]==0:g[r][c]=g[-(r+1)][c];I+=[r];J+=[c]
   if g[-(r+1)][c]==0:g[-(r+1)][c]=g[r][c];I+=[h-(r+1)];J+=[c]
 for r in R(h):
  for c in R(w//2+1):
   if g[r][c]==0:g[r][c]=g[r][-(c+1)];I+=[r];J+=[c]
   if g[r][-(c+1)]==0:g[r][-(c+1)]=g[r][c];I+=[r];J+=[w-(c+1)]
 g=g[min(I):max(I)+1]
 g=[r[min(J):max(J)+1]for r in g]
 return g


# --- Code Golf Solution (Compressed) ---
def q(m):
    return [r[~r.index(0)::-1][:3] for r in m if 0 in r]


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

ContainerContainer = Container[Container]

def identity(
    x: Any
) -> Any:
    """ identity function """
    return x

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

def shape(
    piece: Piece
) -> IntegerTuple:
    """ height and width of grid or patch """
    return (height(piece), width(piece))

def asindices(
    grid: Grid
) -> Indices:
    """ indices of all grid cells """
    return frozenset((i, j) for i in range(len(grid)) for j in range(len(grid[0])))

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

def crop(
    grid: Grid,
    start: IntegerTuple,
    dims: IntegerTuple
) -> Grid:
    """ subgrid specified by start and dimension """
    return tuple(r[start[1]:start[1]+dims[1]] for r in grid[start[0]:start[0]+dims[0]])

def toindices(
    patch: Patch
) -> Indices:
    """ indices of object cells """
    if len(patch) == 0:
        return frozenset()
    if isinstance(next(iter(patch))[1], tuple):
        return frozenset(index for value, index in patch)
    return patch

def dneighbors(
    loc: IntegerTuple
) -> Indices:
    """ directly adjacent indices """
    return frozenset({(loc[0] - 1, loc[1]), (loc[0] + 1, loc[1]), (loc[0], loc[1] - 1), (loc[0], loc[1] + 1)})

def ineighbors(
    loc: IntegerTuple
) -> Indices:
    """ diagonally adjacent indices """
    return frozenset({(loc[0] - 1, loc[1] - 1), (loc[0] - 1, loc[1] + 1), (loc[0] + 1, loc[1] - 1), (loc[0] + 1, loc[1] + 1)})

def neighbors(
    loc: IntegerTuple
) -> Indices:
    """ adjacent indices """
    return dneighbors(loc) | ineighbors(loc)

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

def hmirror(
    piece: Piece
) -> Piece:
    """ mirroring along horizontal """
    if isinstance(piece, tuple):
        return piece[::-1]
    d = ulcorner(piece)[0] + lrcorner(piece)[0]
    if isinstance(next(iter(piece))[1], tuple):
        return frozenset((v, (d - i, j)) for v, (i, j) in piece)
    return frozenset((d - i, j) for i, j in piece)

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

def subgrid(
    patch: Patch,
    grid: Grid
) -> Grid:
    """ smallest subgrid containing object """
    return crop(grid, ulcorner(patch), shape(patch))

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

def generate_9ecd008a(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(1, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 15))
    w = h
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, 8))
    remcols = sample(remcols, numcols)
    canv = canvas(bgc, (h, w))
    nc = unifint(diff_lb, diff_ub, (1, h * w))
    bx = asindices(canv)
    obj = {(choice(remcols), choice(totuple(bx)))}
    for kk in range(nc - 1):
        dns = mapply(neighbors, toindices(obj))
        ch = choice(totuple(bx & dns))
        obj.add((choice(remcols), ch))
        bx = bx - {ch}
    gi = paint(canv, obj)
    tr = sfilter(asobject(dmirror(gi)), lambda cij: cij[1][1] >= cij[1][0])
    gi = paint(gi, tr)
    gi = hconcat(gi, vmirror(gi))
    gi = vconcat(gi, hmirror(gi))
    locidev = unifint(diff_lb, diff_ub, (1, 2*h))
    locjdev = unifint(diff_lb, diff_ub, (1, w))
    loci = 2*h - locidev
    locj = w - locjdev
    loci2 = unifint(diff_lb, diff_ub, (loci, 2*h - 1))
    locj2 = unifint(diff_lb, diff_ub, (locj, w - 1))
    bd = backdrop(frozenset({(loci, locj), (loci2, locj2)}))
    go = subgrid(bd, gi)
    gi = fill(gi, 0, bd)
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Element = Union[Object, Grid]

ZERO = 0

def argmin(
    container: Container,
    compfunc: Callable
) -> Any:
    """ smallest item by custom order """
    return min(container, key=compfunc, default=None)

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

def colorcount(
    element: Element,
    value: Integer
) -> Integer:
    """ number of cells with color """
    if isinstance(element, tuple):
        return sum(row.count(value) for row in element)
    return sum(v == value for v, _ in element)

def ofcolor(
    grid: Grid,
    value: Integer
) -> Indices:
    """ indices of all grid cells with value """
    return frozenset((i, j) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value)

def toobject(
    patch: Patch,
    grid: Grid
) -> Object:
    """ object from patch and grid """
    h, w = len(grid), len(grid[0])
    return frozenset((grid[i][j], (i, j)) for i, j in toindices(patch) if 0 <= i < h and 0 <= j < w)

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_9ecd008a(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = ofcolor(I, ZERO)
    x1 = rbind(colorcount, ZERO)
    x2 = lbind(toobject, x0)
    x3 = compose(x1, x2)
    x4 = vmirror(I)
    x5 = hmirror(I)
    x6 = astuple(x4, x5)
    x7 = argmin(x6, x3)
    x8 = subgrid(x0, x7)
    return x8


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_9ecd008a(inp)
        assert pred == _to_grid(expected), f"{name} failed"

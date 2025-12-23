# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "dc0a314f"
SERIAL = "351"
URL    = "https://arcprize.org/play?task=dc0a314f"

# --- Code Golf Concepts ---
CONCEPTS = [
    "pattern_completion",
    "crop",
]

# --- Example Grids ---
E1_IN = np.array([
    [2, 1, 2, 2, 6, 5, 5, 6, 6, 5, 5, 6, 2, 2, 1, 2],
    [1, 6, 6, 1, 5, 6, 5, 2, 2, 5, 6, 5, 1, 6, 6, 1],
    [2, 6, 1, 6, 5, 5, 5, 2, 2, 5, 5, 5, 6, 1, 6, 2],
    [2, 1, 6, 6, 6, 2, 2, 2, 2, 2, 2, 6, 6, 6, 1, 2],
    [6, 5, 5, 6, 5, 8, 5, 7, 7, 5, 8, 5, 6, 5, 5, 6],
    [5, 6, 5, 2, 8, 8, 5, 8, 8, 3, 3, 3, 3, 3, 6, 5],
    [5, 5, 5, 2, 5, 5, 5, 8, 8, 3, 3, 3, 3, 3, 5, 5],
    [6, 2, 2, 2, 7, 8, 8, 8, 8, 3, 3, 3, 3, 3, 2, 6],
    [6, 2, 2, 2, 7, 8, 8, 8, 8, 3, 3, 3, 3, 3, 2, 6],
    [5, 5, 5, 2, 5, 5, 5, 8, 8, 3, 3, 3, 3, 3, 5, 5],
    [5, 6, 5, 2, 8, 8, 5, 8, 8, 5, 8, 8, 2, 5, 6, 5],
    [6, 5, 5, 6, 5, 8, 5, 7, 7, 5, 8, 5, 6, 5, 5, 6],
    [2, 1, 6, 6, 6, 2, 2, 2, 2, 2, 2, 6, 6, 6, 1, 2],
    [2, 6, 1, 6, 5, 5, 5, 2, 2, 5, 5, 5, 6, 1, 6, 2],
    [1, 6, 6, 1, 5, 6, 5, 2, 2, 5, 6, 5, 1, 6, 6, 1],
    [2, 1, 2, 2, 6, 5, 5, 6, 6, 5, 5, 6, 2, 2, 1, 2],
], dtype=int)

E1_OUT = np.array([
    [5, 8, 8, 2, 5],
    [5, 5, 5, 2, 5],
    [8, 8, 7, 2, 2],
    [8, 8, 7, 2, 2],
    [5, 5, 5, 2, 5],
], dtype=int)

E2_IN = np.array([
    [8, 9, 9, 3, 3, 3, 3, 3, 2, 2, 7, 7, 8, 9, 9, 8],
    [9, 8, 9, 3, 3, 3, 3, 3, 2, 7, 1, 7, 9, 9, 8, 9],
    [9, 9, 8, 3, 3, 3, 3, 3, 7, 2, 7, 2, 2, 8, 9, 9],
    [8, 9, 2, 3, 3, 3, 3, 3, 1, 7, 2, 2, 9, 2, 9, 8],
    [7, 7, 2, 3, 3, 3, 3, 3, 7, 8, 7, 2, 2, 2, 7, 7],
    [7, 1, 7, 2, 7, 2, 7, 7, 7, 7, 2, 7, 2, 7, 1, 7],
    [2, 7, 2, 7, 8, 7, 2, 8, 8, 2, 7, 8, 7, 2, 7, 2],
    [2, 2, 7, 1, 7, 7, 8, 2, 2, 8, 7, 7, 1, 7, 2, 2],
    [2, 2, 7, 1, 7, 7, 8, 2, 2, 8, 7, 7, 1, 7, 2, 2],
    [2, 7, 2, 7, 8, 7, 2, 8, 8, 2, 7, 8, 7, 2, 7, 2],
    [7, 1, 7, 2, 7, 2, 7, 7, 7, 7, 2, 7, 2, 7, 1, 7],
    [7, 7, 2, 2, 2, 7, 8, 7, 7, 8, 7, 2, 2, 2, 7, 7],
    [8, 9, 2, 9, 2, 2, 7, 1, 1, 7, 2, 2, 9, 2, 9, 8],
    [9, 9, 8, 2, 2, 7, 2, 7, 7, 2, 7, 2, 2, 8, 9, 9],
    [9, 8, 9, 9, 7, 1, 7, 2, 2, 7, 1, 7, 9, 9, 8, 9],
    [8, 9, 9, 8, 7, 7, 2, 2, 2, 2, 7, 7, 8, 9, 9, 8],
], dtype=int)

E2_OUT = np.array([
    [8, 7, 7, 2, 2],
    [9, 7, 1, 7, 2],
    [2, 2, 7, 2, 7],
    [9, 2, 2, 7, 1],
    [2, 2, 7, 8, 7],
], dtype=int)

E3_IN = np.array([
    [2, 2, 5, 2, 9, 9, 9, 3, 3, 3, 3, 3, 2, 5, 2, 2],
    [2, 5, 4, 4, 9, 5, 2, 3, 3, 3, 3, 3, 4, 4, 5, 2],
    [5, 4, 5, 4, 9, 2, 5, 3, 3, 3, 3, 3, 4, 5, 4, 5],
    [2, 4, 4, 4, 5, 9, 5, 3, 3, 3, 3, 3, 4, 4, 4, 2],
    [9, 9, 9, 5, 9, 6, 9, 3, 3, 3, 3, 3, 5, 9, 9, 9],
    [9, 5, 2, 9, 6, 6, 9, 9, 9, 9, 6, 6, 9, 2, 5, 9],
    [9, 2, 5, 5, 9, 9, 7, 9, 9, 7, 9, 9, 5, 5, 2, 9],
    [5, 9, 5, 2, 9, 9, 9, 6, 6, 9, 9, 9, 2, 5, 9, 5],
    [5, 9, 5, 2, 9, 9, 9, 6, 6, 9, 9, 9, 2, 5, 9, 5],
    [9, 2, 5, 5, 9, 9, 7, 9, 9, 7, 9, 9, 5, 5, 2, 9],
    [9, 5, 2, 9, 6, 6, 9, 9, 9, 9, 6, 6, 9, 2, 5, 9],
    [9, 9, 9, 5, 9, 6, 9, 9, 9, 9, 6, 9, 5, 9, 9, 9],
    [2, 4, 4, 4, 5, 9, 5, 2, 2, 5, 9, 5, 4, 4, 4, 2],
    [5, 4, 5, 4, 9, 2, 5, 5, 5, 5, 2, 9, 4, 5, 4, 5],
    [2, 5, 4, 4, 9, 5, 2, 9, 9, 2, 5, 9, 4, 4, 5, 2],
    [2, 2, 5, 2, 9, 9, 9, 5, 5, 9, 9, 9, 2, 5, 2, 2],
], dtype=int)

E3_OUT = np.array([
    [5, 5, 9, 9, 9],
    [9, 9, 2, 5, 9],
    [5, 5, 5, 2, 9],
    [2, 2, 5, 9, 5],
    [9, 9, 9, 6, 9],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [5, 5, 2, 5, 2, 5, 5, 5, 5, 5, 5, 2, 5, 2, 5, 5],
    [5, 2, 2, 5, 5, 5, 2, 2, 2, 2, 5, 5, 5, 2, 2, 5],
    [2, 2, 5, 8, 5, 2, 2, 5, 5, 2, 2, 5, 8, 5, 2, 2],
    [5, 5, 8, 5, 5, 2, 5, 5, 5, 5, 2, 5, 5, 8, 5, 5],
    [2, 5, 5, 5, 4, 6, 6, 9, 3, 3, 3, 3, 3, 5, 5, 2],
    [5, 5, 2, 2, 6, 6, 9, 9, 3, 3, 3, 3, 3, 2, 5, 5],
    [5, 2, 2, 5, 6, 9, 6, 9, 3, 3, 3, 3, 3, 2, 2, 5],
    [5, 2, 5, 5, 9, 9, 9, 9, 3, 3, 3, 3, 3, 5, 2, 5],
    [5, 2, 5, 5, 9, 9, 9, 9, 3, 3, 3, 3, 3, 5, 2, 5],
    [5, 2, 2, 5, 6, 9, 6, 9, 9, 6, 9, 6, 5, 2, 2, 5],
    [5, 5, 2, 2, 6, 6, 9, 9, 9, 9, 6, 6, 2, 2, 5, 5],
    [2, 5, 5, 5, 4, 6, 6, 9, 9, 6, 6, 4, 5, 5, 5, 2],
    [5, 5, 8, 5, 5, 2, 5, 5, 5, 5, 2, 5, 5, 8, 5, 5],
    [2, 2, 5, 8, 5, 2, 2, 5, 5, 2, 2, 5, 8, 5, 2, 2],
    [5, 2, 2, 5, 5, 5, 2, 2, 2, 2, 5, 5, 5, 2, 2, 5],
    [5, 5, 2, 5, 2, 5, 5, 5, 5, 5, 5, 2, 5, 2, 5, 5],
], dtype=int)

T_OUT = np.array([
    [9, 6, 6, 4, 5],
    [9, 9, 6, 6, 2],
    [9, 6, 9, 6, 5],
    [9, 9, 9, 9, 5],
    [9, 9, 9, 9, 5],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(g,L=len,R=range):
 h,w,I,J=L(g),L(g[0]),[],[]
 for r in R(h//2+1):
  for c in R(w):
   if g[r][c]==3:g[r][c]=g[-(r+1)][c];I+=[r];J+=[c]
   if g[-(r+1)][c]==3:g[-(r+1)][c]=g[r][c];I+=[h-(r+1)];J+=[c]
 for r in R(h):
  for c in R(w//2+1):
   if g[r][c]==3:g[r][c]=g[r][-(c+1)];I+=[r];J+=[c]
   if g[r][-(c+1)]==3:g[r][-(c+1)]=g[r][c];I+=[r];J+=[w-(c+1)]
 g=g[min(I):max(I)+1]
 g=[r[min(J):max(J)+1]for r in g]
 return g


# --- Code Golf Solution (Compressed) ---
def q(i):
    return [r[:5] for x in [*i] if (r := i.pop()[~[*x, 3].index(3)::-1])]


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

def generate_dc0a314f(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(3, interval(0, 10, 1))
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
    gi = fill(gi, 3, bd)
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
IntegerSet = FrozenSet[Integer]

TupleTuple = Tuple[Tuple]

THREE = 3

NEG_ONE = -1

def maximum(
    container: IntegerSet
) -> Integer:
    """ maximum """
    return max(container, default=0)

def pair(
    a: Tuple,
    b: Tuple
) -> TupleTuple:
    """ zipping of two tuples """
    return tuple(zip(a, b))

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

def papply(
    function: Callable,
    a: Tuple,
    b: Tuple
) -> Tuple:
    """ apply function on two vectors """
    return tuple(function(i, j) for i, j in zip(a, b))

def ofcolor(
    grid: Grid,
    value: Integer
) -> Indices:
    """ indices of all grid cells with value """
    return frozenset((i, j) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value)

def cmirror(
    piece: Piece
) -> Piece:
    """ mirroring along counterdiagonal """
    if isinstance(piece, tuple):
        return tuple(zip(*(r[::-1] for r in piece[::-1])))
    return vmirror(dmirror(vmirror(piece)))

def replace(
    grid: Grid,
    replacee: Integer,
    replacer: Integer
) -> Grid:
    """ color substitution """
    return tuple(tuple(replacer if v == replacee else v for v in r) for r in grid)

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_dc0a314f(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = replace(I, THREE, NEG_ONE)
    x1 = dmirror(x0)
    x2 = papply(pair, x0, x1)
    x3 = lbind(apply, maximum)
    x4 = apply(x3, x2)
    x5 = cmirror(x4)
    x6 = papply(pair, x4, x5)
    x7 = apply(x3, x6)
    x8 = hmirror(x7)
    x9 = papply(pair, x7, x8)
    x10 = apply(x3, x9)
    x11 = vmirror(x10)
    x12 = papply(pair, x11, x10)
    x13 = apply(x3, x12)
    x14 = ofcolor(I, THREE)
    x15 = subgrid(x14, x13)
    return x15


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_dc0a314f(inp)
        assert pred == _to_grid(expected), f"{name} failed"

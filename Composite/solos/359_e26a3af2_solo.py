# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "e26a3af2"
SERIAL = "359"
URL    = "https://arcprize.org/play?task=e26a3af2"

# --- Code Golf Concepts ---
CONCEPTS = [
    "remove_noise",
    "separate_images",
]

# --- Example Grids ---
E1_IN = np.array([
    [1, 1, 1, 1, 1, 8, 8, 8, 3, 3, 3, 3, 2, 2, 2, 8, 2],
    [9, 5, 1, 5, 1, 8, 8, 8, 3, 3, 3, 3, 2, 2, 2, 2, 2],
    [4, 1, 1, 2, 1, 8, 8, 5, 3, 3, 8, 3, 2, 8, 2, 2, 7],
    [1, 1, 1, 1, 1, 8, 8, 2, 3, 3, 3, 3, 2, 2, 2, 2, 2],
    [9, 1, 1, 1, 8, 8, 8, 8, 3, 3, 4, 3, 8, 2, 2, 2, 2],
    [4, 1, 2, 1, 1, 7, 8, 8, 3, 3, 3, 3, 2, 2, 6, 2, 9],
    [1, 1, 1, 1, 9, 8, 8, 8, 9, 3, 3, 3, 4, 2, 6, 2, 2],
    [1, 1, 1, 1, 1, 8, 5, 8, 3, 3, 3, 4, 2, 2, 2, 2, 3],
    [1, 1, 1, 9, 1, 8, 8, 8, 3, 3, 3, 3, 2, 2, 2, 2, 2],
    [6, 1, 1, 8, 1, 5, 8, 8, 4, 3, 3, 3, 6, 4, 2, 2, 7],
    [1, 1, 1, 1, 1, 8, 8, 8, 3, 3, 3, 3, 2, 2, 6, 2, 2],
    [1, 1, 1, 1, 1, 8, 8, 8, 3, 3, 7, 3, 2, 2, 2, 2, 2],
    [1, 2, 1, 4, 1, 8, 8, 8, 3, 3, 3, 3, 2, 9, 2, 1, 2],
], dtype=int)

E1_OUT = np.array([
    [1, 1, 1, 1, 1, 8, 8, 8, 3, 3, 3, 3, 2, 2, 2, 2, 2],
    [1, 1, 1, 1, 1, 8, 8, 8, 3, 3, 3, 3, 2, 2, 2, 2, 2],
    [1, 1, 1, 1, 1, 8, 8, 8, 3, 3, 3, 3, 2, 2, 2, 2, 2],
    [1, 1, 1, 1, 1, 8, 8, 8, 3, 3, 3, 3, 2, 2, 2, 2, 2],
    [1, 1, 1, 1, 1, 8, 8, 8, 3, 3, 3, 3, 2, 2, 2, 2, 2],
    [1, 1, 1, 1, 1, 8, 8, 8, 3, 3, 3, 3, 2, 2, 2, 2, 2],
    [1, 1, 1, 1, 1, 8, 8, 8, 3, 3, 3, 3, 2, 2, 2, 2, 2],
    [1, 1, 1, 1, 1, 8, 8, 8, 3, 3, 3, 3, 2, 2, 2, 2, 2],
    [1, 1, 1, 1, 1, 8, 8, 8, 3, 3, 3, 3, 2, 2, 2, 2, 2],
    [1, 1, 1, 1, 1, 8, 8, 8, 3, 3, 3, 3, 2, 2, 2, 2, 2],
    [1, 1, 1, 1, 1, 8, 8, 8, 3, 3, 3, 3, 2, 2, 2, 2, 2],
    [1, 1, 1, 1, 1, 8, 8, 8, 3, 3, 3, 3, 2, 2, 2, 2, 2],
    [1, 1, 1, 1, 1, 8, 8, 8, 3, 3, 3, 3, 2, 2, 2, 2, 2],
], dtype=int)

E2_IN = np.array([
    [2, 2, 2, 2, 2, 8, 8, 1, 8, 8, 8, 1, 1, 1],
    [2, 2, 8, 2, 2, 8, 8, 8, 8, 8, 8, 1, 1, 1],
    [2, 2, 2, 2, 2, 8, 8, 9, 8, 8, 8, 1, 1, 1],
    [2, 2, 2, 2, 2, 8, 9, 8, 6, 8, 8, 1, 1, 1],
    [2, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 1, 1, 1],
    [2, 2, 2, 2, 2, 8, 8, 8, 8, 8, 5, 1, 1, 1],
    [2, 2, 2, 6, 2, 8, 8, 8, 8, 8, 5, 1, 1, 6],
    [2, 6, 4, 2, 2, 9, 8, 8, 8, 8, 8, 1, 1, 1],
    [2, 2, 2, 2, 2, 6, 8, 7, 8, 8, 8, 1, 1, 2],
    [2, 2, 2, 6, 2, 8, 3, 8, 5, 8, 8, 3, 1, 1],
    [2, 2, 2, 2, 5, 8, 2, 8, 5, 8, 8, 1, 1, 1],
    [2, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 8, 1, 3],
    [2, 8, 2, 2, 2, 8, 8, 8, 8, 3, 8, 9, 1, 1],
], dtype=int)

E2_OUT = np.array([
    [2, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 1, 1, 1],
    [2, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 1, 1, 1],
    [2, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 1, 1, 1],
    [2, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 1, 1, 1],
    [2, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 1, 1, 1],
    [2, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 1, 1, 1],
    [2, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 1, 1, 1],
    [2, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 1, 1, 1],
    [2, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 1, 1, 1],
    [2, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 1, 1, 1],
    [2, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 1, 1, 1],
    [2, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 1, 1, 1],
    [2, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 1, 1, 1],
], dtype=int)

E3_IN = np.array([
    [3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 2, 3, 3, 2, 3, 3, 3, 3, 3],
    [3, 3, 3, 9, 3, 3, 3, 2, 3, 3, 3, 9, 3, 3],
    [3, 3, 4, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3],
    [7, 7, 7, 7, 7, 7, 8, 7, 7, 3, 3, 7, 7, 4],
    [9, 7, 7, 7, 3, 7, 7, 7, 7, 7, 7, 7, 7, 7],
    [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 2],
    [7, 7, 7, 7, 7, 5, 7, 7, 7, 7, 7, 7, 5, 8],
    [7, 7, 7, 7, 7, 7, 3, 7, 7, 7, 7, 2, 7, 7],
    [7, 7, 7, 4, 6, 7, 7, 7, 7, 7, 9, 7, 7, 7],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 3, 8, 8],
    [8, 8, 8, 4, 8, 8, 8, 7, 9, 8, 8, 8, 8, 8],
    [1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 9, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 1, 1, 1],
], dtype=int)

E3_OUT = np.array([
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
    [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
    [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
    [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
    [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
    [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [6, 1, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 4, 1, 1, 9, 1, 1, 1, 1, 5, 1, 1, 1, 1, 1],
    [5, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [7, 2, 2, 2, 2, 6, 2, 9, 2, 2, 4, 2, 4, 2, 2],
    [2, 2, 9, 2, 1, 2, 2, 2, 3, 2, 2, 8, 2, 7, 2],
    [2, 5, 2, 2, 5, 6, 6, 2, 2, 2, 3, 2, 5, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 6, 2, 8, 2, 2],
    [1, 8, 8, 8, 8, 8, 9, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 1, 8, 8, 8, 8, 8, 7, 8, 8, 8, 9],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 5, 8, 8, 8, 1, 8, 8],
    [4, 4, 4, 4, 4, 4, 7, 3, 4, 4, 4, 4, 4, 2, 4],
    [4, 4, 7, 4, 4, 4, 4, 4, 4, 4, 8, 4, 4, 4, 4],
    [3, 3, 1, 9, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [8, 6, 3, 3, 8, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
], dtype=int)

T_OUT = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def X(g):return list(zip(*g[::-1]))
def p(g,L=len,R=range):
 V=0
 if max(g[0].count(i) for i in R(10))-1<L(g[0])/2:V=1;g=X(g)
 h,w=L(g),L(g[0])
 for r in R(h):
  C=sorted([[g[r].count(i),i] for i in R(10)])[-1][1]
  g[r]=[C]*w
 if V:g=X(X(X((g))))
 return [list(r) for r in g]


# --- Code Golf Solution (Compressed) ---
def q(m):
    return [[max((a := (r + c)), key=a.count) for *c, in zip(*m)] for r in m]


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

Piece = Union[Grid, Patch]

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

def toindices(
    patch: Patch
) -> Indices:
    """ indices of object cells """
    if len(patch) == 0:
        return frozenset()
    if isinstance(next(iter(patch))[1], tuple):
        return frozenset(index for value, index in patch)
    return patch

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

def generate_e26a3af2(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    nr = unifint(diff_lb, diff_ub, (1, 10))
    w = unifint(diff_lb, diff_ub, (4, 30))
    scols = sample(cols, nr)
    sgs = [canvas(col, (2, w)) for col in scols]
    numexp = unifint(diff_lb, diff_ub, (0, 30 - nr))
    for k in range(numexp):
        idx = randint(0, nr - 1)
        sgs[idx] = sgs[idx] + sgs[idx][-1:]
    sgs2 = []
    for idx, col in enumerate(scols):
        sg = sgs[idx]
        a, b = shape(sg)
        ub = (a * b) // 2 - 1
        nnoise = unifint(diff_lb, diff_ub, (0, ub))
        inds = totuple(asindices(sg))
        noise = sample(inds, nnoise)
        oc = remove(col, cols)
        noise = frozenset({(choice(oc), ij) for ij in noise})
        sg2 = paint(sg, noise)
        for idxx in [0, -1]:
            while sum([e == col for e in sg2[idxx]]) < w // 2:
                locs = [j for j, e in enumerate(sg2[idxx]) if e != col]
                ch = choice(locs)
                if idxx == 0:
                    sg2 = (sg2[0][:ch] + (col,) + sg2[0][ch+1:],) + sg2[1:]
                else:
                    sg2 = sg2[:-1] + (sg2[-1][:ch] + (col,) + sg2[-1][ch+1:],)
        sgs2.append(sg2)
    gi = tuple(row for sg in sgs2 for row in sg)
    go = tuple(row for sg in sgs for row in sg)
    if choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

ONE = 1

def dedupe(
    iterable: Tuple
) -> Tuple:
    """ remove duplicates """
    return tuple(e for i, e in enumerate(iterable) if iterable.index(e) == i)

def repeat(
    item: Any,
    num: Integer
) -> Tuple:
    """ repetition of item within vector """
    return tuple(item for i in range(num))

def greater(
    a: Integer,
    b: Integer
) -> Boolean:
    """ greater """
    return a > b

def size(
    container: Container
) -> Integer:
    """ cardinality """
    return len(container)

def mostcommon(
    container: Container
) -> Any:
    """ most common item """
    return max(set(container), key=container.count)

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

def apply(
    function: Callable,
    container: Container
) -> Container:
    """ apply function to each item in container """
    return type(container)(function(e) for e in container)

def rot90(
    grid: Grid
) -> Grid:
    """ quarter clockwise rotation """
    return tuple(row for row in zip(*grid[::-1]))

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

def vupscale(
    grid: Grid,
    factor: Integer
) -> Grid:
    """ upscale grid vertically """
    upscaled_grid = tuple()
    for row in grid:
        upscaled_grid = upscaled_grid + tuple(row for num in range(factor))
    return upscaled_grid

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_e26a3af2(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = rot90(I)
    x1 = apply(mostcommon, I)
    x2 = apply(mostcommon, x0)
    x3 = repeat(x1, ONE)
    x4 = repeat(x2, ONE)
    x5 = compose(size, dedupe)
    x6 = x5(x1)
    x7 = x5(x2)
    x8 = greater(x7, x6)
    x9 = branch(x8, height, width)
    x10 = x9(I)
    x11 = rot90(x3)
    x12 = branch(x8, x4, x11)
    x13 = branch(x8, vupscale, hupscale)
    x14 = x13(x12, x10)
    return x14


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_e26a3af2(inp)
        assert pred == _to_grid(expected), f"{name} failed"

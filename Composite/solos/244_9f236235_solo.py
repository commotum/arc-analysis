# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "9f236235"
SERIAL = "244"
URL    = "https://arcprize.org/play?task=9f236235"

# --- Code Golf Concepts ---
CONCEPTS = [
    "detect_grid",
    "summarize",
    "image_reflection",
]

# --- Example Grids ---
E1_IN = np.array([
    [3, 3, 3, 3, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0],
    [3, 3, 3, 3, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0],
    [3, 3, 3, 3, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0],
    [3, 3, 3, 3, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [0, 0, 0, 0, 2, 3, 3, 3, 3, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 3, 3, 3, 3, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 3, 3, 3, 3, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 3, 3, 3, 3, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 3, 3, 3, 3, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 3, 3, 3, 3, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 3, 3, 3, 3, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 3, 3, 3, 3, 2, 0, 0, 0, 0],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [3, 3, 3, 3, 2, 3, 3, 3, 3, 2, 3, 3, 3, 3, 2, 0, 0, 0, 0],
    [3, 3, 3, 3, 2, 3, 3, 3, 3, 2, 3, 3, 3, 3, 2, 0, 0, 0, 0],
    [3, 3, 3, 3, 2, 3, 3, 3, 3, 2, 3, 3, 3, 3, 2, 0, 0, 0, 0],
    [3, 3, 3, 3, 2, 3, 3, 3, 3, 2, 3, 3, 3, 3, 2, 0, 0, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [0, 0, 0, 3],
    [0, 0, 3, 0],
    [0, 3, 0, 0],
    [0, 3, 3, 3],
], dtype=int)

E2_IN = np.array([
    [0, 0, 0, 0, 8, 2, 2, 2, 2, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 2, 2, 2, 2, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 2, 2, 2, 2, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 2, 2, 2, 2, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [2, 2, 2, 2, 8, 1, 1, 1, 1, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [2, 2, 2, 2, 8, 1, 1, 1, 1, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [2, 2, 2, 2, 8, 1, 1, 1, 1, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [2, 2, 2, 2, 8, 1, 1, 1, 1, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 1, 1, 1, 1, 8, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 1, 1, 1, 1, 8, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 1, 1, 1, 1, 8, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 1, 1, 1, 1, 8, 0, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 3, 3, 3, 3],
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 3, 3, 3, 3],
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 3, 3, 3, 3],
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 3, 3, 3, 3],
], dtype=int)

E2_OUT = np.array([
    [0, 0, 2, 0],
    [0, 0, 1, 2],
    [0, 1, 0, 0],
    [3, 0, 0, 0],
], dtype=int)

E3_IN = np.array([
    [0, 0, 0, 2, 8, 8, 8, 2, 0, 0, 0],
    [0, 0, 0, 2, 8, 8, 8, 2, 0, 0, 0],
    [0, 0, 0, 2, 8, 8, 8, 2, 0, 0, 0],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [8, 8, 8, 2, 8, 8, 8, 2, 0, 0, 0],
    [8, 8, 8, 2, 8, 8, 8, 2, 0, 0, 0],
    [8, 8, 8, 2, 8, 8, 8, 2, 0, 0, 0],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [0, 0, 0, 2, 0, 0, 0, 2, 4, 4, 4],
    [0, 0, 0, 2, 0, 0, 0, 2, 4, 4, 4],
    [0, 0, 0, 2, 0, 0, 0, 2, 4, 4, 4],
], dtype=int)

E3_OUT = np.array([
    [0, 8, 0],
    [0, 8, 8],
    [4, 0, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [1, 1, 1, 1, 1, 8, 3, 3, 3, 3, 3, 8, 1, 1, 1, 1, 1, 8, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 8, 3, 3, 3, 3, 3, 8, 1, 1, 1, 1, 1, 8, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 8, 3, 3, 3, 3, 3, 8, 1, 1, 1, 1, 1, 8, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 8, 3, 3, 3, 3, 3, 8, 1, 1, 1, 1, 1, 8, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 8, 3, 3, 3, 3, 3, 8, 1, 1, 1, 1, 1, 8, 1, 1, 1, 1, 1],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 0, 0, 8, 3, 3, 3, 3, 3, 8, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 8, 3, 3, 3, 3, 3, 8, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 8, 3, 3, 3, 3, 3, 8, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 8, 3, 3, 3, 3, 3, 8, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 8, 3, 3, 3, 3, 3, 8, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [2, 2, 2, 2, 2, 8, 3, 3, 3, 3, 3, 8, 0, 0, 0, 0, 0, 8, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 8, 3, 3, 3, 3, 3, 8, 0, 0, 0, 0, 0, 8, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 8, 3, 3, 3, 3, 3, 8, 0, 0, 0, 0, 0, 8, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 8, 3, 3, 3, 3, 3, 8, 0, 0, 0, 0, 0, 8, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 8, 3, 3, 3, 3, 3, 8, 0, 0, 0, 0, 0, 8, 2, 2, 2, 2, 2],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 0, 0, 8, 3, 3, 3, 3, 3, 8, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 8, 3, 3, 3, 3, 3, 8, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 8, 3, 3, 3, 3, 3, 8, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 8, 3, 3, 3, 3, 3, 8, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 8, 3, 3, 3, 3, 3, 8, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0],
], dtype=int)

T_OUT = np.array([
    [1, 1, 3, 1],
    [0, 0, 3, 0],
    [2, 0, 3, 2],
    [0, 0, 3, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(g,V=range):R,C=len(g),len(g[0]);G=[-1]+[i for i in V(R)if len({*g[i]})==1]+[R];z=[-1]+[j for j in V(C)if len({g[i][j]for i in V(R)})==1]+[C];o=[[g[a+1][c+1]for c,d in zip(z,z[1:])if c+1<d-1]for a,b in zip(G,G[1:])if a+1<b-1];return[o[::-1]for o in o]


# --- Code Golf Solution (Compressed) ---
def q(g, w=2):
    return [[0, max, p][w]((g := r), -2) for r in g if g != r][::w]


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

def generate_9f236235(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    numh = unifint(diff_lb, diff_ub, (2, 14))
    numw = unifint(diff_lb, diff_ub, (2, 14))
    h = unifint(diff_lb, diff_ub, (1, 31 // numh - 1))
    w = unifint(diff_lb, diff_ub, (1, 31 // numw - 1))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    frontcol = choice(remcols)
    remcols = remove(frontcol, cols)
    numcols = unifint(diff_lb, diff_ub, (1, min(9, numh * numw)))
    ccols = sample(remcols, numcols)
    numcells = unifint(diff_lb, diff_ub, (1, numh * numw))
    cands = asindices(canvas(-1, (numh, numw)))
    inds = asindices(canvas(-1, (h, w)))
    locs = sample(totuple(cands), numcells)
    gi = canvas(frontcol, (h * numh + numh - 1, w * numw + numw - 1))
    go = canvas(bgc, (numh, numw))
    for cand in cands:
        a, b = cand
        plcd = shift(inds, (a * (h + 1), b * (w + 1)))
        col = choice(remcols) if cand in locs else bgc
        gi = fill(gi, col, plcd)
        go = fill(go, col, {cand})
    go = vmirror(go)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

ContainerContainer = Container[Container]

F = False

T = True

def identity(
    x: Any
) -> Any:
    """ identity function """
    return x

def order(
    container: Container,
    compfunc: Callable
) -> Tuple:
    """ order container by custom key """
    return tuple(sorted(container, key=compfunc))

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

def objects(
    grid: Grid,
    univalued: Boolean,
    diagonal: Boolean,
    without_bg: Boolean
) -> Objects:
    """ objects occurring on the grid """
    bg = mostcolor(grid) if without_bg else None
    objs = set()
    occupied = set()
    h, w = len(grid), len(grid[0])
    unvisited = asindices(grid)
    diagfun = neighbors if diagonal else dneighbors
    for loc in unvisited:
        if loc in occupied:
            continue
        val = grid[loc[0]][loc[1]]
        if val == bg:
            continue
        obj = {(val, loc)}
        cands = {loc}
        while len(cands) > 0:
            neighborhood = set()
            for cand in cands:
                v = grid[cand[0]][cand[1]]
                if (val == v) if univalued else (v != bg):
                    obj.add((v, cand))
                    occupied.add(cand)
                    neighborhood |= {
                        (i, j) for i, j in diagfun(cand) if 0 <= i < h and 0 <= j < w
                    }
            cands = neighborhood - occupied
        objs.add(frozenset(obj))
    return frozenset(objs)

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

def hconcat(
    a: Grid,
    b: Grid
) -> Grid:
    """ concatenate two grids horizontally """
    return tuple(i + j for i, j in zip(a, b))

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

def verify_9f236235(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = frontiers(I)
    x1 = merge(x0)
    x2 = color(x1)
    x3 = shape(I)
    x4 = canvas(x2, x3)
    x5 = hconcat(I, x4)
    x6 = objects(x5, T, F, T)
    x7 = apply(uppermost, x6)
    x8 = order(x7, identity)
    x9 = lbind(sfilter, x6)
    x10 = lbind(matcher, uppermost)
    x11 = compose(x9, x10)
    x12 = lbind(apply, color)
    x13 = rbind(order, leftmost)
    x14 = chain(x12, x13, x11)
    x15 = apply(x14, x8)
    x16 = vmirror(x15)
    return x16


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_9f236235(inp)
        assert pred == _to_grid(expected), f"{name} failed"

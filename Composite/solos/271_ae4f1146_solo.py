# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "ae4f1146"
SERIAL = "271"
URL    = "https://arcprize.org/play?task=ae4f1146"

# --- Code Golf Concepts ---
CONCEPTS = [
    "separate_images",
    "count_tiles",
    "crop",
]

# --- Example Grids ---
E1_IN = np.array([
    [8, 8, 8, 0, 0, 0, 0, 0, 0],
    [1, 8, 8, 0, 8, 1, 8, 0, 0],
    [8, 8, 8, 0, 1, 1, 8, 0, 0],
    [0, 0, 0, 0, 8, 8, 8, 0, 0],
    [0, 8, 8, 1, 0, 0, 0, 0, 0],
    [0, 8, 8, 8, 0, 0, 8, 1, 8],
    [0, 8, 1, 8, 0, 0, 1, 8, 1],
    [0, 0, 0, 0, 0, 0, 1, 8, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [8, 1, 8],
    [1, 8, 1],
    [1, 8, 1],
], dtype=int)

E2_IN = np.array([
    [0, 8, 8, 1, 0, 0, 0, 0, 0],
    [0, 8, 1, 8, 0, 8, 1, 8, 0],
    [0, 8, 8, 8, 0, 1, 8, 8, 0],
    [0, 0, 0, 0, 0, 8, 8, 1, 0],
    [0, 0, 8, 1, 8, 0, 0, 0, 0],
    [0, 0, 1, 1, 8, 0, 0, 0, 0],
    [0, 0, 8, 8, 1, 0, 8, 8, 8],
    [0, 0, 0, 0, 0, 0, 8, 8, 8],
    [0, 0, 0, 0, 0, 0, 1, 8, 8],
], dtype=int)

E2_OUT = np.array([
    [8, 1, 8],
    [1, 1, 8],
    [8, 8, 1],
], dtype=int)

E3_IN = np.array([
    [0, 0, 0, 0, 8, 8, 8, 0, 0],
    [8, 8, 8, 0, 8, 8, 8, 0, 0],
    [8, 8, 8, 0, 1, 8, 8, 0, 0],
    [8, 8, 8, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 8, 1, 8],
    [8, 1, 8, 0, 0, 0, 1, 1, 8],
    [8, 8, 1, 0, 0, 0, 1, 8, 1],
    [1, 8, 8, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [8, 1, 8],
    [1, 1, 8],
    [1, 8, 1],
], dtype=int)

E4_IN = np.array([
    [0, 0, 1, 1, 8, 0, 0, 0, 0],
    [0, 0, 8, 8, 1, 0, 8, 1, 1],
    [0, 0, 1, 1, 8, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 8, 1, 8],
    [8, 8, 8, 0, 0, 0, 0, 0, 0],
    [8, 8, 1, 0, 8, 1, 8, 0, 0],
    [1, 8, 8, 0, 1, 8, 8, 0, 0],
    [0, 0, 0, 0, 8, 8, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E4_OUT = np.array([
    [8, 1, 1],
    [1, 1, 1],
    [8, 1, 8],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [8, 8, 8, 0, 0, 0, 8, 1, 8],
    [8, 8, 8, 0, 0, 0, 1, 8, 1],
    [1, 8, 8, 0, 0, 0, 8, 1, 8],
    [0, 0, 0, 8, 1, 8, 0, 0, 0],
    [0, 0, 0, 8, 8, 1, 0, 0, 0],
    [0, 0, 0, 1, 8, 8, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 8],
    [0, 0, 0, 0, 0, 0, 8, 1, 1],
    [0, 0, 0, 0, 0, 0, 1, 1, 8],
], dtype=int)

T_OUT = np.array([
    [1, 1, 8],
    [8, 1, 1],
    [1, 1, 8],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(g,L=len,R=range):
 h,w=L(g),L(g[0])
 Z,z=[],0
 for r in R(h-2):
  for c in R(w-2):
   C=g[r][c:c+3]+g[r+1][c:c+3]+g[r+2][c:c+3]
   y=C.count(1)+(C.count(8)/10)
   if y>z:Z=C[:];z=y
 return [Z[:3],Z[3:6],Z[6:]]


# --- Code Golf Solution (Compressed) ---
exec(f"p=lambda g:max([str(g).count('1'),g]{'for*g,in map(zip,g,g[1:],g[2:])'*2})[1]")
# ----------------------------------------------------------------
# cgi, oxjam

def q(*args, **kwargs):
    return p(*args, **kwargs)


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

def difference(
    a: Container,
    b: Container
) -> Container:
    """ difference """
    return type(a)(e for e in a if e not in b)

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

def canvas(
    value: Integer,
    dimensions: IntegerTuple
) -> Grid:
    """ grid construction """
    return tuple(tuple(value for j in range(dimensions[1])) for i in range(dimensions[0]))

def outbox(
    patch: Patch
) -> Indices:
    """ outbox for patch """
    ai, aj = uppermost(patch) - 1, leftmost(patch) - 1
    bi, bj = lowermost(patch) + 1, rightmost(patch) + 1
    si, sj = min(ai, bi), min(aj, bj)
    ei, ej = max(ai, bi), max(aj, bj)
    vlines = {(i, sj) for i in range(si, ei + 1)} | {(i, ej) for i in range(si, ei + 1)}
    hlines = {(si, j) for j in range(sj, ej + 1)} | {(ei, j) for j in range(sj, ej + 1)}
    return frozenset(vlines | hlines)

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

def generate_ae4f1146(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(1, interval(0, 10, 1))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    fgc = choice(remcols)
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    dh = unifint(diff_lb, diff_ub, (2, h // 3))
    dw = unifint(diff_lb, diff_ub, (2, w // 3))
    num = unifint(diff_lb, diff_ub, (1, (h * w) // (2 * dh * dw)))
    cards = interval(0, dh * dw, 1)
    ccards = sorted(sample(cards, min(num, len(cards))))
    sgs = []
    c1 = canvas(fgc, (dh, dw))
    inds = totuple(asindices(c1))
    for card in ccards:
        x = sample(inds, card)
        x1 = fill(c1, 1, x)
        sgs.append(asobject(x1))
    go = paint(c1, sgs[-1])
    gi = canvas(bgc, (h, w))
    inds2 = asindices(canvas(bgc, (h - dh, w - dw)))
    maxtr = 10
    for sg in sgs[::-1]:
        if len(inds2) == 0:
            break
        loc = choice(totuple(inds2))
        plcd = shift(sg, loc)
        tr = 0    
        while (not toindices(plcd).issubset(inds2)) and tr < maxtr:
            loc = choice(totuple(inds2))
            plcd = shift(sg, loc)
            tr += 1
        if tr < maxtr:
            inds2 = difference(inds2, toindices(plcd) | outbox(plcd))
            gi = paint(gi, plcd)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

ONE = 1

F = False

T = True

def argmax(
    container: Container,
    compfunc: Callable
) -> Any:
    """ largest item by custom order """
    return max(container, key=compfunc, default=None)

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

def shape(
    piece: Piece
) -> IntegerTuple:
    """ height and width of grid or patch """
    return (height(piece), width(piece))

def colorcount(
    element: Element,
    value: Integer
) -> Integer:
    """ number of cells with color """
    if isinstance(element, tuple):
        return sum(row.count(value) for row in element)
    return sum(v == value for v, _ in element)

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

def toobject(
    patch: Patch,
    grid: Grid
) -> Object:
    """ object from patch and grid """
    h, w = len(grid), len(grid[0])
    return frozenset((grid[i][j], (i, j)) for i, j in toindices(patch) if 0 <= i < h and 0 <= j < w)

def subgrid(
    patch: Patch,
    grid: Grid
) -> Grid:
    """ smallest subgrid containing object """
    return crop(grid, ulcorner(patch), shape(patch))

def box(
    patch: Patch
) -> Indices:
    """ outline of patch """
    if len(patch) == 0:
        return patch
    ai, aj = ulcorner(patch)
    bi, bj = lrcorner(patch)
    si, sj = min(ai, bi), min(aj, bj)
    ei, ej = max(ai, bi), max(aj, bj)
    vlines = {(i, sj) for i in range(si, ei + 1)} | {(i, ej) for i in range(si, ei + 1)}
    hlines = {(si, j) for j in range(sj, ej + 1)} | {(ei, j) for j in range(sj, ej + 1)}
    return frozenset(vlines | hlines)

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_ae4f1146(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = asindices(I)
    x1 = box(x0)
    x2 = toobject(x1, I)
    x3 = mostcolor(x2)
    x4 = objects(I, F, F, T)
    x5 = rbind(colorcount, ONE)
    x6 = argmax(x4, x5)
    x7 = subgrid(x6, I)
    return x7


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("E4", E4_IN, E4_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_ae4f1146(inp)
        assert pred == _to_grid(expected), f"{name} failed"

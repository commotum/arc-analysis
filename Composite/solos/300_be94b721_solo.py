# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "be94b721"
SERIAL = "300"
URL    = "https://arcprize.org/play?task=be94b721"

# --- Code Golf Concepts ---
CONCEPTS = [
    "separate_shapes",
    "count_tiles",
    "take_maximum",
    "crop",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 2, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0],
    [0, 0, 2, 0, 0, 0, 3, 0, 0, 1, 0, 0, 0],
    [0, 0, 2, 2, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    [0, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [2, 2, 0],
    [0, 2, 0],
    [0, 2, 2],
    [2, 2, 2],
], dtype=int)

E2_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 6, 6, 0],
    [0, 3, 0, 0, 4, 4, 0, 0, 6, 0],
    [3, 3, 3, 0, 4, 4, 0, 0, 0, 0],
    [0, 3, 0, 0, 4, 4, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [4, 4],
    [4, 4],
    [4, 4],
], dtype=int)

E3_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 8, 8, 8, 0, 0, 0, 0, 7, 7, 0],
    [0, 0, 8, 0, 0, 0, 2, 0, 0, 7, 0],
    [0, 8, 8, 0, 0, 2, 2, 0, 0, 7, 0],
    [0, 8, 8, 0, 0, 0, 2, 0, 0, 7, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [8, 8, 8],
    [0, 8, 0],
    [8, 8, 0],
    [8, 8, 0],
], dtype=int)

E4_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 7, 0, 0, 2, 2, 2],
    [0, 0, 0, 7, 7, 0, 0, 2, 0],
    [0, 0, 0, 0, 7, 0, 2, 2, 2],
    [8, 8, 8, 0, 0, 0, 0, 0, 0],
    [0, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E4_OUT = np.array([
    [2, 2, 2],
    [0, 2, 0],
    [2, 2, 2],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 4, 0, 3, 3, 3, 0, 0, 0],
    [0, 4, 0, 3, 3, 3, 0, 0, 0],
    [0, 0, 0, 3, 0, 3, 0, 0, 0],
    [0, 0, 0, 3, 0, 3, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 6, 6],
    [0, 5, 5, 5, 0, 0, 6, 6, 6],
    [0, 5, 5, 0, 0, 0, 6, 6, 0],
], dtype=int)

T_OUT = np.array([
    [3, 3, 3],
    [3, 3, 3],
    [3, 0, 3],
    [3, 0, 3],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
from collections import*
def p(m,K=enumerate):
	a=[(i,j)for(i,r)in K(m)for(j,v)in K(r)if v]
	if not a:return[]
	v=Counter(m[i][j]for(i,j)in a).most_common(1)[0][0];x=[(i,j)for(i,j)in a if m[i][j]==v];h,b=min(i for(i,_)in x),min(j for(_,j)in x);c,g=max(i for(i,_)in x)+1,max(j for(_,j)in x)+1;return[m[i][b:g]for i in range(h,c)]


# --- Code Golf Solution (Compressed) ---
def q(g, *G):
    return [r for *r, in zip(*(G or p(g, *g))) if max(range(1, 10), key=sum(g, g).count) in r]


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, uniform

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

Piece = Union[Grid, Patch]

ContainerContainer = Container[Container]

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

def normalize(
    patch: Patch
) -> Patch:
    """ moves upper left corner to origin """
    if len(patch) == 0:
        return patch
    return shift(patch, (-uppermost(patch), -leftmost(patch)))

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

def generate_be94b721(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    no = unifint(diff_lb, diff_ub, (3, max(3, (h * w) // 16)))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    c = canvas(bgc, (h, w))
    nc = unifint(diff_lb, diff_ub, (no+1, max(no+1, 2*no)))
    inds = asindices(c)
    ch = choice(totuple(inds))
    shp = {ch}
    inds = remove(ch, inds)
    for k in range(nc - 1):
        shp.add(choice(totuple((inds - shp) & mapply(dneighbors, shp))))
    inds = (inds - shp) - mapply(neighbors, shp)
    trgc = choice(remcols)
    gi = fill(c, trgc, shp)
    go = fill(canvas(bgc, shape(shp)), trgc, normalize(shp))
    for k in range(no):
        if len(inds) == 0:
            break
        ch = choice(totuple(inds))
        shp = {ch}
        nc2 = unifint(diff_lb, diff_ub, (1, nc - 1))
        for k in range(nc2 - 1):
            cands = totuple((inds - shp) & mapply(dneighbors, shp))
            if len(cands) == 0:
                break
            shp.add(choice(cands))
        col = choice(remcols)
        gi = fill(gi, col, shp)
        inds = (inds - shp) - mapply(neighbors, shp)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

F = False

T = True

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

def mostcolor(
    element: Element
) -> Integer:
    """ most common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return max(set(values), key=values.count)

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

def color(
    obj: Object
) -> Integer:
    """ color of object """
    return next(iter(obj))[0]

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

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_be94b721(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = objects(I, T, F, F)
    x1 = argmax(x0, size)
    x2 = color(x1)
    x3 = remove(x1, x0)
    x4 = argmax(x3, size)
    x5 = shape(x4)
    x6 = canvas(x2, x5)
    x7 = normalize(x4)
    x8 = paint(x6, x7)
    return x8


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("E4", E4_IN, E4_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_be94b721(inp)
        assert pred == _to_grid(expected), f"{name} failed"

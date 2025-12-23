# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "aedd82e4"
SERIAL = "272"
URL    = "https://arcprize.org/play?task=aedd82e4"

# --- Code Golf Concepts ---
CONCEPTS = [
    "recoloring",
    "separate_shapes",
    "count_tiles",
    "take_minimum",
    "associate_colors_to_bools",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 2, 2],
    [0, 2, 2],
    [2, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [0, 2, 2],
    [0, 2, 2],
    [1, 0, 0],
], dtype=int)

E2_IN = np.array([
    [2, 2, 2, 0],
    [0, 2, 0, 0],
    [0, 0, 0, 2],
    [0, 2, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [2, 2, 2, 0],
    [0, 2, 0, 0],
    [0, 0, 0, 1],
    [0, 1, 0, 0],
], dtype=int)

E3_IN = np.array([
    [2, 2, 0, 0],
    [0, 2, 0, 0],
    [2, 2, 0, 2],
    [0, 0, 0, 0],
    [0, 2, 2, 2],
], dtype=int)

E3_OUT = np.array([
    [2, 2, 0, 0],
    [0, 2, 0, 0],
    [2, 2, 0, 1],
    [0, 0, 0, 0],
    [0, 2, 2, 2],
], dtype=int)

E4_IN = np.array([
    [2, 2, 0],
    [2, 0, 2],
    [0, 2, 0],
], dtype=int)

E4_OUT = np.array([
    [2, 2, 0],
    [2, 0, 1],
    [0, 1, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [2, 2, 0, 2],
    [0, 2, 0, 0],
    [0, 0, 2, 0],
    [2, 0, 0, 0],
    [0, 0, 2, 2],
], dtype=int)

T_OUT = np.array([
    [2, 2, 0, 1],
    [0, 2, 0, 0],
    [0, 0, 1, 0],
    [1, 0, 0, 0],
    [0, 0, 2, 2],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(g):h,w=len(g),len(g[0]);return[[1if g[i][j]and all(g[i+a][j+b]==0for a,b in[(-1,0),(1,0),(0,-1),(0,1)]if 0<=i+a<h and 0<=j+b<w)else g[i][j]for j in range(w)]for i in range(h)]


# --- Code Golf Solution (Compressed) ---
def q(i, *w):
    return i * 0 != 0 and [*map(p, i, [i * 2] + i, i[1:] + [i * 2], *w)] or ~(2 in w) * i % 3


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import sample, uniform

Boolean = bool

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Objects = FrozenSet[Object]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

Element = Union[Object, Grid]

ContainerContainer = Container[Container]

F = False

T = True

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

def mostcolor(
    element: Element
) -> Integer:
    """ most common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return max(set(values), key=values.count)

def sizefilter(
    container: Container,
    n: Integer
) -> FrozenSet:
    """ filter items by size """
    return frozenset(item for item in container if len(item) == n)

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

def generate_aedd82e4(diff_lb: float, diff_ub: float) -> dict:
    colopts = remove(1, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (1, 30))
    w = unifint(diff_lb, diff_ub, (1, 30))
    bgc = 0
    remcols = remove(bgc, colopts)
    c = canvas(bgc, (h, w))
    card_bounds = (0, max(0, (h * w) // 2 - 1))
    num = unifint(diff_lb, diff_ub, card_bounds)
    numcols = unifint(diff_lb, diff_ub, (0, min(8, num)))
    inds = totuple(asindices(c))
    chosinds = sample(inds, num)
    choscols = sample(remcols, numcols)
    locs = interval(0, len(chosinds), 1)
    choslocs = sample(locs, numcols)
    gi = canvas(bgc, (h, w))
    for col, endidx in zip(choscols, sorted(choslocs)[::-1]):
        gi = fill(gi, col, chosinds[:endidx])
    objs = objects(gi, F, F, T)
    res = merge(sizefilter(objs, 1))
    go = fill(gi, 1, res)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Piece = Union[Grid, Patch]

ZERO = 0

ONE = 1

def flip(
    b: Boolean
) -> Boolean:
    """ logical not """
    return not b

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

def matcher(
    function: Callable,
    target: Any
) -> Callable:
    """ construction of equality function """
    return lambda x: function(x) == target

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

def color(
    obj: Object
) -> Integer:
    """ color of object """
    return next(iter(obj))[0]

def hconcat(
    a: Grid,
    b: Grid
) -> Grid:
    """ concatenate two grids horizontally """
    return tuple(i + j for i, j in zip(a, b))

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_aedd82e4(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = shape(I)
    x1 = canvas(ZERO, x0)
    x2 = hconcat(I, x1)
    x3 = objects(x2, F, F, T)
    x4 = matcher(color, ZERO)
    x5 = compose(flip, x4)
    x6 = sfilter(x3, x5)
    x7 = sizefilter(x6, ONE)
    x8 = merge(x7)
    x9 = fill(I, ONE, x8)
    return x9


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("E4", E4_IN, E4_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_aedd82e4(inp)
        assert pred == _to_grid(expected), f"{name} failed"

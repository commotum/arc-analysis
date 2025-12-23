# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "bda2d7a6"
SERIAL = "298"
URL    = "https://arcprize.org/play?task=bda2d7a6"

# --- Code Golf Concepts ---
CONCEPTS = [
    "recoloring",
    "pairwise_analogy",
    "pattern_modification",
    "color_permutation",
]

# --- Example Grids ---
E1_IN = np.array([
    [3, 3, 3, 3, 3, 3],
    [3, 2, 2, 2, 2, 3],
    [3, 2, 0, 0, 2, 3],
    [3, 2, 0, 0, 2, 3],
    [3, 2, 2, 2, 2, 3],
    [3, 3, 3, 3, 3, 3],
], dtype=int)

E1_OUT = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 3, 3, 3, 3, 0],
    [0, 3, 2, 2, 3, 0],
    [0, 3, 2, 2, 3, 0],
    [0, 3, 3, 3, 3, 0],
    [0, 0, 0, 0, 0, 0],
], dtype=int)

E2_IN = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 7, 7, 7, 7, 0],
    [0, 7, 6, 6, 7, 0],
    [0, 7, 6, 6, 7, 0],
    [0, 7, 7, 7, 7, 0],
    [0, 0, 0, 0, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [6, 6, 6, 6, 6, 6],
    [6, 0, 0, 0, 0, 6],
    [6, 0, 7, 7, 0, 6],
    [6, 0, 7, 7, 0, 6],
    [6, 0, 0, 0, 0, 6],
    [6, 6, 6, 6, 6, 6],
], dtype=int)

E3_IN = np.array([
    [8, 8, 8, 8, 8, 8, 8, 8],
    [8, 0, 0, 0, 0, 0, 0, 8],
    [8, 0, 5, 5, 5, 5, 0, 8],
    [8, 0, 5, 8, 8, 5, 0, 8],
    [8, 0, 5, 8, 8, 5, 0, 8],
    [8, 0, 5, 5, 5, 5, 0, 8],
    [8, 0, 0, 0, 0, 0, 0, 8],
    [8, 8, 8, 8, 8, 8, 8, 8],
], dtype=int)

E3_OUT = np.array([
    [5, 5, 5, 5, 5, 5, 5, 5],
    [5, 8, 8, 8, 8, 8, 8, 5],
    [5, 8, 0, 0, 0, 0, 8, 5],
    [5, 8, 0, 5, 5, 0, 8, 5],
    [5, 8, 0, 5, 5, 0, 8, 5],
    [5, 8, 0, 0, 0, 0, 8, 5],
    [5, 8, 8, 8, 8, 8, 8, 5],
    [5, 5, 5, 5, 5, 5, 5, 5],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [9, 9, 9, 9, 9, 9],
    [9, 0, 0, 0, 0, 9],
    [9, 0, 1, 1, 0, 9],
    [9, 0, 1, 1, 0, 9],
    [9, 0, 0, 0, 0, 9],
    [9, 9, 9, 9, 9, 9],
], dtype=int)

T_OUT = np.array([
    [1, 1, 1, 1, 1, 1],
    [1, 9, 9, 9, 9, 1],
    [1, 9, 0, 0, 9, 1],
    [1, 9, 0, 0, 9, 1],
    [1, 9, 9, 9, 9, 1],
    [1, 1, 1, 1, 1, 1],
], dtype=int)

T2_IN = np.array([
    [3, 3, 3, 3, 3, 3, 3, 3],
    [3, 7, 7, 7, 7, 7, 7, 3],
    [3, 7, 6, 6, 6, 6, 7, 3],
    [3, 7, 6, 3, 3, 6, 7, 3],
    [3, 7, 6, 3, 3, 6, 7, 3],
    [3, 7, 6, 6, 6, 6, 7, 3],
    [3, 7, 7, 7, 7, 7, 7, 3],
    [3, 3, 3, 3, 3, 3, 3, 3],
], dtype=int)

T2_OUT = np.array([
    [6, 6, 6, 6, 6, 6, 6, 6],
    [6, 3, 3, 3, 3, 3, 3, 6],
    [6, 3, 7, 7, 7, 7, 3, 6],
    [6, 3, 7, 6, 6, 7, 3, 6],
    [6, 3, 7, 6, 6, 7, 3, 6],
    [6, 3, 7, 7, 7, 7, 3, 6],
    [6, 3, 3, 3, 3, 3, 3, 6],
    [6, 6, 6, 6, 6, 6, 6, 6],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):A=len(j)//2;c=[j[i][i]for i in range(A)];E={c[i]:c[i-1]for i in range(A)};return[[E[i]for i in r]for r in j]


# --- Code Golf Solution (Compressed) ---
def q(g):
    return [[g[2][-r.index(v) | 2] for v in r] for r in g]


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, sample, uniform

Boolean = bool

Integer = int

IntegerTuple = Tuple[Integer, Integer]

IntegerSet = FrozenSet[Integer]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Objects = FrozenSet[Object]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

ContainerContainer = Container[Container]

F = False

T = True

def combine(
    a: Container,
    b: Container
) -> Container:
    """ union """
    return type(a)((*a, *b))

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

def maximum(
    container: IntegerSet
) -> Integer:
    """ maximum """
    return max(container, default=0)

def interval(
    start: Integer,
    stop: Integer,
    step: Integer
) -> Tuple:
    """ range """
    return tuple(range(start, stop, step))

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

def papply(
    function: Callable,
    a: Tuple,
    b: Tuple
) -> Tuple:
    """ apply function on two vectors """
    return tuple(function(i, j) for i, j in zip(a, b))

def mpapply(
    function: Callable,
    a: Tuple,
    b: Tuple
) -> Tuple:
    """ apply function on two vectors and merge """
    return merge(papply(function, a, b))

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

def recolor(
    value: Integer,
    patch: Patch
) -> Object:
    """ recolor patch """
    return frozenset((value, index) for index in toindices(patch))

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

def generate_bda2d7a6(diff_lb: float, diff_ub: float) -> dict:
    colopts = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 14))
    w = unifint(diff_lb, diff_ub, (2, 14))
    ncols = unifint(diff_lb, diff_ub, (2, 10))
    cols = sample(colopts, ncols)
    colord = [choice(cols) for j in range(min(h, w))]
    shp = (h*2, w*2)
    gi = canvas(0, shp)
    for idx, (ci, co) in enumerate(zip(colord, colord[-1:] + colord[:-1])):
        ulc = (idx, idx)
        lrc = (h*2 - 1 - idx, w*2 - 1 - idx)
        bx = box(frozenset({ulc, lrc}))
        gi = fill(gi, ci, bx)
    I = gi
    objso = order(objects(I, T, F, F), compose(maximum, shape))
    if color(objso[0]) == color(objso[-1]):
        objso = (combine(objso[0], objso[-1]),) + objso[1:-1]
    res = mpapply(recolor, apply(color, objso), (objso[-1],) + objso[:-1])
    go = paint(gi, res)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
ONE = 1

def equality(
    a: Any,
    b: Any
) -> Boolean:
    """ equality """
    return a == b

def repeat(
    item: Any,
    num: Integer
) -> Tuple:
    """ repetition of item within vector """
    return tuple(item for i in range(num))

def first(
    container: Container
) -> Any:
    """ first item of container """
    return next(iter(container))

def last(
    container: Container
) -> Any:
    """ last item of container """
    return max(enumerate(container))[1]

def remove(
    value: Any,
    container: Container
) -> Container:
    """ remove item from container """
    return type(container)(e for e in container if e != value)

def branch(
    condition: Boolean,
    if_value: Any,
    else_value: Any
) -> Any:
    """ if else branching """
    return if_value if condition else else_value

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_bda2d7a6(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = objects(I, T, F, F)
    x1 = compose(maximum, shape)
    x2 = order(x0, x1)
    x3 = first(x2)
    x4 = last(x2)
    x5 = color(x3)
    x6 = color(x4)
    x7 = equality(x5, x6)
    x8 = combine(x3, x4)
    x9 = repeat(x8, ONE)
    x10 = remove(x3, x2)
    x11 = remove(x4, x10)
    x12 = combine(x9, x11)
    x13 = branch(x7, x12, x2)
    x14 = apply(color, x13)
    x15 = last(x13)
    x16 = remove(x15, x13)
    x17 = repeat(x15, ONE)
    x18 = combine(x17, x16)
    x19 = mpapply(recolor, x14, x18)
    x20 = paint(I, x19)
    return x20


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
        ("T2", T2_IN, T2_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_bda2d7a6(inp)
        assert pred == _to_grid(expected), f"{name} failed"

# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "e3497940"
SERIAL = "360"
URL    = "https://arcprize.org/play?task=e3497940"

# --- Code Golf Concepts ---
CONCEPTS = [
    "detect_wall",
    "separate_images",
    "image_reflection",
    "image_juxtaposition",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 0, 0, 5, 0, 0, 0, 0],
    [0, 0, 0, 0, 5, 0, 0, 0, 0],
    [0, 0, 0, 4, 5, 0, 0, 0, 0],
    [0, 0, 0, 4, 5, 4, 4, 0, 0],
    [0, 0, 3, 3, 5, 0, 0, 0, 0],
    [0, 0, 0, 3, 5, 0, 0, 0, 0],
    [0, 0, 0, 3, 5, 3, 3, 3, 0],
    [0, 0, 0, 3, 5, 0, 0, 0, 0],
    [0, 0, 0, 0, 5, 0, 0, 0, 0],
    [0, 0, 0, 0, 5, 0, 0, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 4],
    [0, 0, 4, 4],
    [0, 0, 3, 3],
    [0, 0, 0, 3],
    [0, 3, 3, 3],
    [0, 0, 0, 3],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
], dtype=int)

E2_IN = np.array([
    [0, 0, 0, 0, 5, 0, 0, 0, 0],
    [0, 0, 0, 2, 5, 0, 0, 0, 0],
    [0, 0, 0, 2, 5, 2, 6, 0, 0],
    [0, 0, 0, 2, 5, 0, 0, 0, 0],
    [0, 0, 0, 2, 5, 2, 2, 2, 0],
    [0, 0, 6, 6, 5, 6, 0, 0, 0],
    [0, 0, 0, 2, 5, 0, 0, 0, 0],
    [0, 2, 2, 0, 5, 2, 0, 0, 0],
    [0, 0, 0, 2, 5, 0, 0, 0, 0],
    [0, 0, 0, 0, 5, 0, 0, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 2],
    [0, 0, 6, 2],
    [0, 0, 0, 2],
    [0, 2, 2, 2],
    [0, 0, 6, 6],
    [0, 0, 0, 2],
    [0, 2, 2, 2],
    [0, 0, 0, 2],
    [0, 0, 0, 0],
], dtype=int)

E3_IN = np.array([
    [0, 0, 0, 0, 5, 0, 0, 0, 0],
    [0, 0, 0, 0, 5, 7, 0, 0, 0],
    [0, 0, 0, 8, 5, 0, 0, 0, 0],
    [0, 0, 0, 8, 5, 0, 0, 0, 0],
    [0, 7, 8, 8, 5, 0, 0, 0, 0],
    [0, 0, 0, 0, 5, 8, 8, 0, 0],
    [0, 0, 0, 8, 5, 0, 0, 0, 0],
    [0, 0, 0, 8, 5, 0, 0, 0, 0],
    [0, 0, 0, 0, 5, 8, 7, 0, 0],
    [0, 0, 0, 0, 5, 0, 0, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 7],
    [0, 0, 0, 8],
    [0, 0, 0, 8],
    [0, 7, 8, 8],
    [0, 0, 8, 8],
    [0, 0, 0, 8],
    [0, 0, 0, 8],
    [0, 0, 7, 8],
    [0, 0, 0, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 0, 0, 5, 0, 0, 0, 0],
    [0, 0, 0, 1, 5, 0, 0, 0, 0],
    [0, 0, 0, 1, 5, 1, 0, 0, 0],
    [0, 1, 1, 1, 5, 1, 1, 1, 6],
    [0, 0, 0, 6, 5, 6, 6, 0, 0],
    [0, 0, 0, 0, 5, 1, 1, 1, 0],
    [0, 0, 0, 1, 5, 0, 0, 0, 0],
    [0, 0, 0, 1, 5, 1, 6, 0, 0],
    [0, 0, 0, 0, 5, 6, 0, 0, 0],
    [0, 0, 0, 0, 5, 0, 0, 0, 0],
], dtype=int)

T_OUT = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [6, 1, 1, 1],
    [0, 0, 6, 6],
    [0, 1, 1, 1],
    [0, 0, 0, 1],
    [0, 0, 6, 1],
    [0, 0, 0, 6],
    [0, 0, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(g):
    return [[g[i][j] or g[i][8 - j] if g[i][j] * g[i][8 - j] == 0 else g[i][j] for j in range(4)] for i in range(len(g))]


# --- Code Golf Solution (Compressed) ---
def q(m):
    return [[*map(max, r, r[:4:-1])] for r in m]


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

def generate_e3497940(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (3, 14))
    bgc, barc = sample(cols, 2)
    remcols = remove(barc, remove(bgc, cols))
    ncols = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, ncols)
    nlinesocc = unifint(diff_lb, diff_ub, (1, h))
    lopts = interval(0, h, 1)
    linesocc = sample(lopts, nlinesocc)
    rs = canvas(bgc, (h, w))
    ls = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    for idx in linesocc:
        j = unifint(diff_lb, diff_ub, (1, w - 1))
        obj = [(choice(ccols), (idx, jj)) for jj in range(j)]
        go = paint(go, obj)
        slen = randint(1, j)
        obj2 = obj[:slen]
        if choice((True, False)):
            obj, obj2 = obj2, obj
        rs = paint(rs, obj)
        ls = paint(ls, obj2)
    gi = hconcat(hconcat(vmirror(ls), canvas(barc, (h, 1))), rs)
    go = vmirror(go)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

ContainerContainer = Container[Container]

F = False

T = True

def flip(
    b: Boolean
) -> Boolean:
    """ logical not """
    return not b

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

def mostcolor(
    element: Element
) -> Integer:
    """ most common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return max(set(values), key=values.count)

def width(
    piece: Piece
) -> Integer:
    """ width of grid or patch """
    if len(piece) == 0:
        return 0
    if isinstance(piece, tuple):
        return len(piece[0])
    return rightmost(piece) - leftmost(piece) + 1

def asindices(
    grid: Grid
) -> Indices:
    """ indices of all grid cells """
    return frozenset((i, j) for i in range(len(grid)) for j in range(len(grid[0])))

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

def hsplit(
    grid: Grid,
    n: Integer
) -> Tuple:
    """ split grid horizontally """
    h, w = len(grid), len(grid[0]) // n
    offset = len(grid[0]) % n != 0
    return tuple(crop(grid, (0, w * i + i * offset), (h, w)) for i in range(n))

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

def verify_e3497940(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = lefthalf(I)
    x1 = righthalf(I)
    x2 = vmirror(x1)
    x3 = width(I)
    x4 = hsplit(I, x3)
    x5 = first(x4)
    x6 = mostcolor(x5)
    x7 = objects(x2, T, F, F)
    x8 = matcher(color, x6)
    x9 = compose(flip, x8)
    x10 = sfilter(x7, x9)
    x11 = merge(x10)
    x12 = paint(x0, x11)
    return x12


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_e3497940(inp)
        assert pred == _to_grid(expected), f"{name} failed"

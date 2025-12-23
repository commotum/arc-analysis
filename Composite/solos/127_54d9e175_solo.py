# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "54d9e175"
SERIAL = "127"
URL    = "https://arcprize.org/play?task=54d9e175"

# --- Code Golf Concepts ---
CONCEPTS = [
    "detect_grid",
    "separate_images",
    "associate_images_to_images",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 1, 0, 5, 0, 2, 0, 5, 0, 1, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [6, 6, 6, 5, 7, 7, 7, 5, 6, 6, 6],
    [6, 6, 6, 5, 7, 7, 7, 5, 6, 6, 6],
    [6, 6, 6, 5, 7, 7, 7, 5, 6, 6, 6],
], dtype=int)

E2_IN = np.array([
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 2, 0, 5, 0, 3, 0, 5, 0, 1, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [7, 7, 7, 5, 8, 8, 8, 5, 6, 6, 6],
    [7, 7, 7, 5, 8, 8, 8, 5, 6, 6, 6],
    [7, 7, 7, 5, 8, 8, 8, 5, 6, 6, 6],
], dtype=int)

E3_IN = np.array([
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 3, 0, 5, 0, 1, 0, 5, 0, 4, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [8, 8, 8, 5, 6, 6, 6, 5, 9, 9, 9],
    [8, 8, 8, 5, 6, 6, 6, 5, 9, 9, 9],
    [8, 8, 8, 5, 6, 6, 6, 5, 9, 9, 9],
], dtype=int)

E4_IN = np.array([
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 4, 0, 5, 0, 1, 0, 5, 0, 2, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 2, 0, 5, 0, 3, 0, 5, 0, 4, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
], dtype=int)

E4_OUT = np.array([
    [9, 9, 9, 5, 6, 6, 6, 5, 7, 7, 7],
    [9, 9, 9, 5, 6, 6, 6, 5, 7, 7, 7],
    [9, 9, 9, 5, 6, 6, 6, 5, 7, 7, 7],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [7, 7, 7, 5, 8, 8, 8, 5, 9, 9, 9],
    [7, 7, 7, 5, 8, 8, 8, 5, 9, 9, 9],
    [7, 7, 7, 5, 8, 8, 8, 5, 9, 9, 9],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 2, 0, 5, 0, 3, 0, 5, 0, 4, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 1, 0, 5, 0, 1, 0, 5, 0, 3, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
], dtype=int)

T_OUT = np.array([
    [7, 7, 7, 5, 8, 8, 8, 5, 9, 9, 9],
    [7, 7, 7, 5, 8, 8, 8, 5, 9, 9, 9],
    [7, 7, 7, 5, 8, 8, 8, 5, 9, 9, 9],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [6, 6, 6, 5, 6, 6, 6, 5, 8, 8, 8],
    [6, 6, 6, 5, 6, 6, 6, 5, 8, 8, 8],
    [6, 6, 6, 5, 6, 6, 6, 5, 8, 8, 8],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(g):
 R=range;Z=[r[:]for r in g];h,w=len(g),len(g[0])
 for r in R(1,h,4):
  for c in R(1,w,4):
   C=g[r][c]+5
   for y in R(3):
    for x in R(3):Z[r-1+y][c-1+x]=C
 return Z


# --- Code Golf Solution (Compressed) ---
def q(g):
    return g * -1 and g + 5 or (g and [p(g[1])] * 3 + g[3:4] + p(g[4:]))


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

def totuple(
    container: FrozenSet
) -> Tuple:
    """ conversion to tuple """
    return tuple(container)

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

def generate_54d9e175(diff_lb: float, diff_ub: float) -> dict:
    cols = (0, 5)
    h = unifint(diff_lb, diff_ub, (2, 5))
    w = unifint(diff_lb, diff_ub, (2, 5))
    nh = unifint(diff_lb, diff_ub, (1, 31 // (h + 1)))
    nw = unifint(diff_lb, diff_ub, (1 if nh > 1 else 2, 31 // (w + 1)))
    fullh = (h + 1) * nh - 1
    fullw = (w + 1) * nw - 1
    linc, bgc = sample(cols, 2)
    gi = canvas(linc, (fullh, fullw))
    go = canvas(linc, (fullh, fullw))
    obj = asindices(canvas(bgc, (h, w)))
    for a in range(nh):
        for b in range(nw):
            plcd = shift(obj, (a * (h + 1), b * (w + 1)))
            icol = randint(1, 4)
            ocol = icol + 5
            gi = fill(gi, bgc, plcd)
            go = fill(go, ocol, plcd)
            dot = choice(totuple(plcd))
            gi = fill(gi, icol, {dot})
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Numerical = Union[Integer, IntegerTuple]

IntegerSet = FrozenSet[Integer]

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

ContainerContainer = Container[Container]

ZERO = 0

FIVE = 5

F = False

T = True

def merge(
    containers: ContainerContainer
) -> Container:
    """ merging """
    return type(containers)(e for c in containers for e in c)

def increment(
    x: Numerical
) -> Numerical:
    """ incrementing """
    return x + 1 if isinstance(x, int) else (x[0] + 1, x[1] + 1)

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

def power(
    function: Callable,
    n: Integer
) -> Callable:
    """ power of function """
    if n == 1:
        return function
    return compose(function, power(function, n - 1))

def fork(
    outer: Callable,
    a: Callable,
    b: Callable
) -> Callable:
    """ creates a wrapper function """
    return lambda x: outer(a(x), b(x))

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

def mostcolor(
    element: Element
) -> Integer:
    """ most common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return max(set(values), key=values.count)

def leastcolor(
    element: Element
) -> Integer:
    """ least common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return min(set(values), key=values.count)

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

def ulcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of upper left corner """
    return tuple(map(min, zip(*toindices(patch))))

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

def palette(
    element: Element
) -> IntegerSet:
    """ colors occurring in object or grid """
    if isinstance(element, tuple):
        return frozenset({v for r in element for v in r})
    return frozenset({v for v, _ in element})

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

def verify_54d9e175(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = frontiers(I)
    x1 = merge(x0)
    x2 = leastcolor(x1)
    x3 = shape(I)
    x4 = canvas(x2, x3)
    x5 = hconcat(I, x4)
    x6 = objects(x5, F, F, T)
    x7 = power(increment, FIVE)
    x8 = lbind(remove, FIVE)
    x9 = lbind(remove, ZERO)
    x10 = chain(x8, x9, palette)
    x11 = chain(x7, first, x10)
    x12 = fork(recolor, x11, toindices)
    x13 = mapply(x12, x6)
    x14 = paint(I, x13)
    return x14


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("E4", E4_IN, E4_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_54d9e175(inp)
        assert pred == _to_grid(expected), f"{name} failed"

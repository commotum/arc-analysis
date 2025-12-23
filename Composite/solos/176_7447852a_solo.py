# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "7447852a"
SERIAL = "176"
URL    = "https://arcprize.org/play?task=7447852a"

# --- Code Golf Concepts ---
CONCEPTS = [
    "pattern_expansion",
    "pairwise_analogy",
]

# --- Example Grids ---
E1_IN = np.array([
    [2, 0, 0, 0, 2, 0, 0, 0, 2, 0],
    [0, 2, 0, 2, 0, 2, 0, 2, 0, 2],
    [0, 0, 2, 0, 0, 0, 2, 0, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [2, 0, 0, 0, 2, 4, 4, 4, 2, 0],
    [4, 2, 0, 2, 0, 2, 4, 2, 0, 2],
    [4, 4, 2, 0, 0, 0, 2, 0, 0, 0],
], dtype=int)

E2_IN = np.array([
    [2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0],
    [0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0],
    [0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2],
], dtype=int)

E2_OUT = np.array([
    [2, 0, 0, 0, 2, 4, 4, 4, 2, 0, 0, 0, 2, 0, 0],
    [4, 2, 0, 2, 0, 2, 4, 2, 0, 2, 0, 2, 4, 2, 0],
    [4, 4, 2, 0, 0, 0, 2, 0, 0, 0, 2, 4, 4, 4, 2],
], dtype=int)

E3_IN = np.array([
    [2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0],
    [0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2],
    [0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [2, 0, 0, 0, 2, 4, 4, 4, 2, 0, 0, 0, 2, 0, 0, 0, 2, 4],
    [4, 2, 0, 2, 0, 2, 4, 2, 0, 2, 0, 2, 4, 2, 0, 2, 0, 2],
    [4, 4, 2, 0, 0, 0, 2, 0, 0, 0, 2, 4, 4, 4, 2, 0, 0, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2],
    [0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0],
    [0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0],
], dtype=int)

T_OUT = np.array([
    [2, 0, 0, 0, 2, 4, 4, 4, 2, 0, 0, 0, 2, 0, 0, 0, 2, 4, 4, 4, 2, 0, 0, 0, 2],
    [4, 2, 0, 2, 0, 2, 4, 2, 0, 2, 0, 2, 4, 2, 0, 2, 0, 2, 4, 2, 0, 2, 0, 2, 4],
    [4, 4, 2, 0, 0, 0, 2, 0, 0, 0, 2, 4, 4, 4, 2, 0, 0, 0, 2, 0, 0, 0, 2, 4, 4],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
 A,c,E=j;k=6,4,0,0,0,1,3,1,0,0,0,4
 for W in range(len(A)):
  l=k[W%12]
  if l&1:A[W]=4
  if l&2:c[W]=4
  if l&4:E[W]=4
 return j


# --- Code Golf Solution (Compressed) ---
def q(g, r=5):
    return [[x | ~r * (r := (r - x)) % 6 for x in s] * (r := 1) for s in g]


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, sample, uniform

Boolean = bool

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Numerical = Union[Integer, IntegerTuple]

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

def add(
    a: Numerical,
    b: Numerical
) -> Numerical:
    """ addition """
    if isinstance(a, int) and isinstance(b, int):
        return a + b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] + b[0], a[1] + b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a + b[0], a + b[1])
    return (a[0] + b, a[1] + b)

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

def colorfilter(
    objs: Objects,
    value: Integer
) -> Objects:
    """ filter objects by color """
    return frozenset(obj for obj in objs if next(iter(obj))[0] == value)

def asindices(
    grid: Grid
) -> Indices:
    """ indices of all grid cells """
    return frozenset((i, j) for i in range(len(grid)) for j in range(len(grid[0])))

def ofcolor(
    grid: Grid,
    value: Integer
) -> Indices:
    """ indices of all grid cells with value """
    return frozenset((i, j) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value)

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

def leftmost(
    patch: Patch
) -> Integer:
    """ column index of leftmost occupied cell """
    return min(j for i, j in toindices(patch))

def rot90(
    grid: Grid
) -> Grid:
    """ quarter clockwise rotation """
    return tuple(row for row in zip(*grid[::-1]))

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

def generate_7447852a(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(4, interval(0, 10, 1))
    w = unifint(diff_lb, diff_ub, (2, 8))
    h = unifint(diff_lb, diff_ub, (w+1, 30))
    bgc, linc = sample(cols, 2)
    remcols = remove(bgc, remove(linc, cols))
    c = canvas(bgc, (h, w))
    sp = (h - 1, 0)
    gi = fill(c, linc, {sp})
    direc = 1
    while True:
        sp = add(sp, (-1, direc))
        if sp[1] == w - 1 or sp[1] == 0:
            direc *= -1
        gi2 = fill(gi, linc, {sp})
        if gi2 == gi:
            break
        gi = gi2
    gi = rot90(gi)
    objs = objects(gi, T, F, F)
    inds = ofcolor(gi, bgc)
    numcols = unifint(diff_lb, diff_ub, (1, 7))
    ccols = sample(remcols, numcols)
    ncells = unifint(diff_lb, diff_ub, (0, len(inds)))
    locs = sample(totuple(inds), ncells)
    obj = {(choice(ccols), ij) for ij in locs}
    gi = paint(gi, obj)
    go = tuple(e for e in gi)
    objs = order(colorfilter(objs, bgc), leftmost)
    objs = merge(set(objs[0::3]))
    go = fill(go, 4, objs)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Piece = Union[Grid, Patch]

TupleTuple = Tuple[Tuple]

ZERO = 0

ONE = 1

THREE = 3

FOUR = 4

ORIGIN = (0, 0)

def multiply(
    a: Numerical,
    b: Numerical
) -> Numerical:
    """ multiplication """
    if isinstance(a, int) and isinstance(b, int):
        return a * b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] * b[0], a[1] * b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a * b[0], a * b[1])
    return (a[0] * b, a[1] * b)

def divide(
    a: Numerical,
    b: Numerical
) -> Numerical:
    """ floor division """
    if isinstance(a, int) and isinstance(b, int):
        return a // b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] // b[0], a[1] // b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a // b[0], a // b[1])
    return (a[0] // b, a[1] // b)

def equality(
    a: Any,
    b: Any
) -> Boolean:
    """ equality """
    return a == b

def size(
    container: Container
) -> Integer:
    """ cardinality """
    return len(container)

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

def last(
    container: Container
) -> Any:
    """ last item of container """
    return max(enumerate(container))[1]

def pair(
    a: Tuple,
    b: Tuple
) -> TupleTuple:
    """ zipping of two tuples """
    return tuple(zip(a, b))

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

def rightmost(
    patch: Patch
) -> Integer:
    """ column index of rightmost occupied cell """
    return max(j for i, j in toindices(patch))

def centerofmass(
    patch: Patch
) -> IntegerTuple:
    """ center of mass """
    return tuple(map(lambda x: sum(x) // len(patch), zip(*toindices(patch))))

def hconcat(
    a: Grid,
    b: Grid
) -> Grid:
    """ concatenate two grids horizontally """
    return tuple(i + j for i, j in zip(a, b))

def index(
    grid: Grid,
    loc: IntegerTuple
) -> Integer:
    """ color at location """
    i, j = loc
    h, w = len(grid), len(grid[0])
    if not (0 <= i < h and 0 <= j < w):
        return None
    return grid[loc[0]][loc[1]]

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_7447852a(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = index(I, ORIGIN)
    x1 = shape(I)
    x2 = canvas(x0, x1)
    x3 = hconcat(I, x2)
    x4 = objects(x3, F, F, T)
    x5 = compose(last, centerofmass)
    x6 = order(x4, x5)
    x7 = size(x6)
    x8 = interval(ZERO, x7, ONE)
    x9 = pair(x6, x8)
    x10 = rbind(multiply, THREE)
    x11 = rbind(divide, THREE)
    x12 = chain(x10, x11, last)
    x13 = fork(equality, last, x12)
    x14 = sfilter(x9, x13)
    x15 = mapply(first, x14)
    x16 = fill(I, FOUR, x15)
    return x16


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_7447852a(inp)
        assert pred == _to_grid(expected), f"{name} failed"

# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "caa06a1f"
SERIAL = "313"
URL    = "https://arcprize.org/play?task=caa06a1f"

# --- Code Golf Concepts ---
CONCEPTS = [
    "pattern_expansion",
    "image_filling",
]

# --- Example Grids ---
E1_IN = np.array([
    [6, 7, 6, 7, 6, 7, 6, 3, 3, 3, 3],
    [7, 6, 7, 6, 7, 6, 7, 3, 3, 3, 3],
    [6, 7, 6, 7, 6, 7, 6, 3, 3, 3, 3],
    [7, 6, 7, 6, 7, 6, 7, 3, 3, 3, 3],
    [6, 7, 6, 7, 6, 7, 6, 3, 3, 3, 3],
    [7, 6, 7, 6, 7, 6, 7, 3, 3, 3, 3],
    [6, 7, 6, 7, 6, 7, 6, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
], dtype=int)

E1_OUT = np.array([
    [7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7],
    [6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6],
    [7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7],
    [6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6],
    [7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7],
    [6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6],
    [7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7],
    [6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6],
    [7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7],
    [6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6],
    [7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7],
], dtype=int)

E2_IN = np.array([
    [6, 3, 6, 3, 6, 3, 6, 1],
    [3, 6, 3, 6, 3, 6, 3, 1],
    [6, 3, 6, 3, 6, 3, 6, 1],
    [3, 6, 3, 6, 3, 6, 3, 1],
    [6, 3, 6, 3, 6, 3, 6, 1],
    [3, 6, 3, 6, 3, 6, 3, 1],
    [6, 3, 6, 3, 6, 3, 6, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
], dtype=int)

E2_OUT = np.array([
    [3, 6, 3, 6, 3, 6, 3, 6],
    [6, 3, 6, 3, 6, 3, 6, 3],
    [3, 6, 3, 6, 3, 6, 3, 6],
    [6, 3, 6, 3, 6, 3, 6, 3],
    [3, 6, 3, 6, 3, 6, 3, 6],
    [6, 3, 6, 3, 6, 3, 6, 3],
    [3, 6, 3, 6, 3, 6, 3, 6],
    [6, 3, 6, 3, 6, 3, 6, 3],
], dtype=int)

E3_IN = np.array([
    [5, 4, 5, 4, 5, 6],
    [4, 5, 4, 5, 4, 6],
    [5, 4, 5, 4, 5, 6],
    [4, 5, 4, 5, 4, 6],
    [5, 4, 5, 4, 5, 6],
    [6, 6, 6, 6, 6, 6],
], dtype=int)

E3_OUT = np.array([
    [4, 5, 4, 5, 4, 5],
    [5, 4, 5, 4, 5, 4],
    [4, 5, 4, 5, 4, 5],
    [5, 4, 5, 4, 5, 4],
    [4, 5, 4, 5, 4, 5],
    [5, 4, 5, 4, 5, 4],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 3, 3, 3, 3, 3, 3],
    [5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 3, 3, 3, 3, 3, 3],
    [8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 3, 3, 3, 3, 3, 3],
    [5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 3, 3, 3, 3, 3, 3],
    [8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 3, 3, 3, 3, 3, 3],
    [5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 3, 3, 3, 3, 3, 3],
    [8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 3, 3, 3, 3, 3, 3],
    [5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 3, 3, 3, 3, 3, 3],
    [8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 3, 3, 3, 3, 3, 3],
    [5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 3, 3, 3, 3, 3, 3],
    [8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 3, 3, 3, 3, 3, 3],
    [5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
], dtype=int)

T_OUT = np.array([
    [5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8],
    [7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5],
    [5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8],
    [7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5],
    [5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8],
    [7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5],
    [5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8],
    [7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5],
    [5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8],
    [7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5],
    [5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8],
    [7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5],
    [5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8],
    [7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5],
    [5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8],
    [7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5],
    [5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8],
    [7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5, 7, 8, 5],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(g,r=range,l=len):
 n=l(g);q=l(set(g[0]))-1;p=l({i[0]for i in g})-1
 for x in g:x[:]=(x[:q]*((l(x)-1)//q+1))[:l(x)]
 for i in r(n):g[i]=[g[i%p][j]for j in r(n)]
 return[[dict(zip(g[0],g[0][1:]))[y]for y in r]for r in g]


# --- Code Golf Solution (Compressed) ---
def q(g, u=[]):
    return g * -1 * -1 or [*map(p, g[u > []:2 - -len(u) // 11] * 10, g)]


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

def identity(
    x: Any
) -> Any:
    """ identity function """
    return x

def difference(
    a: Container,
    b: Container
) -> Container:
    """ difference """
    return type(a)(e for e in a if e not in b)

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

def connect(
    a: IntegerTuple,
    b: IntegerTuple
) -> Indices:
    """ line between two points """
    ai, aj = a
    bi, bj = b
    si = min(ai, bi)
    ei = max(ai, bi) + 1
    sj = min(aj, bj)
    ej = max(aj, bj) + 1
    if ai == bi:
        return frozenset((ai, j) for j in range(sj, ej))
    elif aj == bj:
        return frozenset((i, aj) for i in range(si, ei))
    elif bi - ai == bj - aj:
        return frozenset((i, j) for i, j in zip(range(si, ei), range(sj, ej)))
    elif bi - ai == aj - bj:
        return frozenset((i, j) for i, j in zip(range(si, ei), range(ej - 1, sj - 1, -1)))
    return frozenset()

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

def generate_caa06a1f(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    vp = unifint(diff_lb, diff_ub, (2, h//2-1))
    hp = unifint(diff_lb, diff_ub, (2, w//2-1))
    bgc = choice(cols)
    numc = unifint(diff_lb, diff_ub, (2, min(8, max(2, hp * vp))))
    remcols = remove(bgc, cols)
    ccols = sample(remcols, numc)
    remcols = difference(remcols, ccols)
    tric = choice(remcols)
    obj = {(choice(ccols), ij) for ij in asindices(canvas(-1, (vp, hp)))}
    go = canvas(bgc, (h, w))
    gi = canvas(bgc, (h, w))
    for a in range(-vp, h+1, vp):
        for b in range(-hp, w+1, hp):
            go = paint(go, shift(obj, (a, b + 1)))
    for a in range(-vp, h+1, vp):
        for b in range(-hp, w+1, hp):
            gi = paint(gi, shift(obj, (a, b)))
    ioffs = unifint(diff_lb, diff_ub, (1, h - 2 * vp))
    joffs = unifint(diff_lb, diff_ub, (1, w - 2 * hp))
    for a in range(ioffs):
        gi = fill(gi, tric, connect((a, 0), (a, w - 1)))
    for b in range(joffs):
        gi = fill(gi, tric, connect((0, b), (h - 1, b)))
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Numerical = Union[Integer, IntegerTuple]

Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

ContainerContainer = Container[Container]

ONE = 1

TWO = 2

ORIGIN = (0, 0)

DOWN = (1, 0)

RIGHT = (0, 1)

UP = (-1, 0)

LEFT = (0, -1)

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

def subtract(
    a: Numerical,
    b: Numerical
) -> Numerical:
    """ subtraction """
    if isinstance(a, int) and isinstance(b, int):
        return a - b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] - b[0], a[1] - b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a - b[0], a - b[1])
    return (a[0] - b, a[1] - b)

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

def invert(
    n: Numerical
) -> Numerical:
    """ inversion with respect to addition """
    return -n if isinstance(n, int) else (-n[0], -n[1])

def flip(
    b: Boolean
) -> Boolean:
    """ logical not """
    return not b

def equality(
    a: Any,
    b: Any
) -> Boolean:
    """ equality """
    return a == b

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

def product(
    a: Container,
    b: Container
) -> FrozenSet:
    """ cartesian product """
    return frozenset((i, j) for j in b for i in a)

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

def ulcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of upper left corner """
    return tuple(map(min, zip(*toindices(patch))))

def urcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of upper right corner """
    return tuple(map(lambda ix: {0: min, 1: max}[ix[0]](ix[1]), enumerate(zip(*toindices(patch)))))

def llcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of lower left corner """
    return tuple(map(lambda ix: {0: max, 1: min}[ix[0]](ix[1]), enumerate(zip(*toindices(patch)))))

def lrcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of lower right corner """
    return tuple(map(max, zip(*toindices(patch))))

def normalize(
    patch: Patch
) -> Patch:
    """ moves upper left corner to origin """
    if len(patch) == 0:
        return patch
    return shift(patch, (-uppermost(patch), -leftmost(patch)))

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

def toobject(
    patch: Patch,
    grid: Grid
) -> Object:
    """ object from patch and grid """
    h, w = len(grid), len(grid[0])
    return frozenset((grid[i][j], (i, j)) for i, j in toindices(patch) if 0 <= i < h and 0 <= j < w)

def asobject(
    grid: Grid
) -> Object:
    """ conversion of grid to object """
    return frozenset((v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r))

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

def hperiod(
    obj: Object
) -> Integer:
    """ horizontal periodicity """
    normalized = normalize(obj)
    w = width(normalized)
    for p in range(1, w):
        offsetted = shift(normalized, (0, -p))
        pruned = frozenset({(c, (i, j)) for c, (i, j) in offsetted if j >= 0})
        if pruned.issubset(normalized):
            return p
    return w

def vperiod(
    obj: Object
) -> Integer:
    """ vertical periodicity """
    normalized = normalize(obj)
    h = height(normalized)
    for p in range(1, h):
        offsetted = shift(normalized, (-p, 0))
        pruned = frozenset({(c, (i, j)) for c, (i, j) in offsetted if i >= 0})
        if pruned.issubset(normalized):
            return p
    return h

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_caa06a1f(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = asindices(I)
    x1 = box(x0)
    x2 = toobject(x1, I)
    x3 = mostcolor(x2)
    x4 = asobject(I)
    x5 = matcher(first, x3)
    x6 = compose(flip, x5)
    x7 = sfilter(x4, x6)
    x8 = hperiod(x7)
    x9 = vperiod(x7)
    x10 = width(I)
    x11 = width(x7)
    x12 = subtract(x10, x11)
    x13 = add(x12, TWO)
    x14 = height(I)
    x15 = height(x7)
    x16 = subtract(x14, x15)
    x17 = add(x16, TWO)
    x18 = rbind(multiply, x8)
    x19 = invert(x13)
    x20 = interval(x19, x13, ONE)
    x21 = apply(x18, x20)
    x22 = rbind(multiply, x9)
    x23 = invert(x17)
    x24 = interval(x23, x17, ONE)
    x25 = apply(x22, x24)
    x26 = product(x25, x21)
    x27 = lbind(shift, x7)
    x28 = mapply(x27, x26)
    x29 = index(I, ORIGIN)
    x30 = equality(x29, x3)
    x31 = flip(x30)
    x32 = asindices(I)
    x33 = urcorner(x32)
    x34 = index(I, x33)
    x35 = equality(x34, x3)
    x36 = flip(x35)
    x37 = asindices(I)
    x38 = lrcorner(x37)
    x39 = index(I, x38)
    x40 = equality(x39, x3)
    x41 = flip(x40)
    x42 = asindices(I)
    x43 = llcorner(x42)
    x44 = index(I, x43)
    x45 = equality(x44, x3)
    x46 = flip(x45)
    x47 = multiply(x31, LEFT)
    x48 = multiply(x36, UP)
    x49 = add(x47, x48)
    x50 = multiply(x41, RIGHT)
    x51 = multiply(x46, DOWN)
    x52 = add(x50, x51)
    x53 = add(x49, x52)
    x54 = shift(x28, x53)
    x55 = paint(I, x54)
    return x55


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_caa06a1f(inp)
        assert pred == _to_grid(expected), f"{name} failed"

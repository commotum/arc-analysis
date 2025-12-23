# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "508bd3b6"
SERIAL = "119"
URL    = "https://arcprize.org/play?task=508bd3b6"

# --- Code Golf Concepts ---
CONCEPTS = [
    "draw_line_from_point",
    "direction_guessing",
    "pattern_reflection",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2],
    [0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 2, 2],
    [0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 2, 2],
], dtype=int)

E1_OUT = np.array([
    [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 2, 2],
    [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 2, 2],
    [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 2, 2],
    [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 2, 2],
    [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 2, 2],
    [0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 2, 2],
    [0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 2, 2],
], dtype=int)

E2_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
], dtype=int)

E2_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
    [0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 3, 0],
    [0, 0, 0, 3, 0, 0, 0, 0, 0, 3, 0, 0],
    [0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0],
    [0, 0, 0, 0, 0, 3, 0, 3, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
], dtype=int)

E3_IN = np.array([
    [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0],
    [2, 2, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [2, 2, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [2, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0],
    [2, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
    [2, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
    [2, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
    [2, 2, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0],
    [2, 2, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [2, 2, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 0, 8, 0, 0, 0, 0, 2, 2, 2, 2],
    [0, 0, 0, 0, 8, 0, 0, 0, 2, 2, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2],
], dtype=int)

T_OUT = np.array([
    [0, 0, 0, 8, 0, 0, 0, 0, 2, 2, 2, 2],
    [0, 0, 0, 0, 8, 0, 0, 0, 2, 2, 2, 2],
    [0, 0, 0, 0, 0, 3, 0, 0, 2, 2, 2, 2],
    [0, 0, 0, 0, 0, 0, 3, 0, 2, 2, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 3, 2, 2, 2, 2],
    [0, 0, 0, 0, 0, 0, 3, 0, 2, 2, 2, 2],
    [0, 0, 0, 0, 0, 3, 0, 0, 2, 2, 2, 2],
    [0, 0, 0, 0, 3, 0, 0, 0, 2, 2, 2, 2],
    [0, 0, 0, 3, 0, 0, 0, 0, 2, 2, 2, 2],
    [0, 0, 3, 0, 0, 0, 0, 0, 2, 2, 2, 2],
    [0, 3, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2],
    [3, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(*args, **kwargs):
    raise NotImplementedError("Barnacles solution not available for 119")


# --- Code Golf Solution (Compressed) ---
def q(g):
    return exec('g[::-1]=zip(*eval(re.sub("0(?=.{40}[38].{40}[238])","3",str(g))));' * 40) or g


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

def ofcolor(
    grid: Grid,
    value: Integer
) -> Indices:
    """ indices of all grid cells with value """
    return frozenset((i, j) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value)

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

def hmirror(
    piece: Piece
) -> Piece:
    """ mirroring along horizontal """
    if isinstance(piece, tuple):
        return piece[::-1]
    d = ulcorner(piece)[0] + lrcorner(piece)[0]
    if isinstance(next(iter(piece))[1], tuple):
        return frozenset((v, (d - i, j)) for v, (i, j) in piece)
    return frozenset((d - i, j) for i, j in piece)

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

def cmirror(
    piece: Piece
) -> Piece:
    """ mirroring along counterdiagonal """
    if isinstance(piece, tuple):
        return tuple(zip(*(r[::-1] for r in piece[::-1])))
    return vmirror(dmirror(vmirror(piece)))

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

def shoot(
    start: IntegerTuple,
    direction: IntegerTuple
) -> Indices:
    """ line from starting point and direction """
    return connect(start, (start[0] + 42 * direction[0], start[1] + 42 * direction[1]))

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

def generate_508bd3b6(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(3, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (h, 30))
    barh = unifint(diff_lb, diff_ub, (1, h // 2))
    barloci = unifint(diff_lb, diff_ub, (2, h - barh))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    barc = choice(remcols)
    remcols = remove(barc, remcols)
    linc = choice(remcols)
    gi = canvas(bgc, (h, w))
    for j in range(barloci, barloci + barh):
        gi = fill(gi, barc, connect((j, 0), (j, w - 1)))
    dotlociinv = unifint(diff_lb, diff_ub, (0, barloci - 1))
    dotloci = min(max(0, barloci - 2 - dotlociinv), barloci - 1)
    ln1 = shoot((dotloci, 0), (1, 1))
    ofbgc = ofcolor(gi, bgc)
    ln1 = sfilter(ln1 & ofbgc, lambda ij: ij[0] < barloci)
    ln1 = order(ln1, first)
    ln2 = shoot(ln1[-1], (-1, 1))
    ln2 = sfilter(ln2 & ofbgc, lambda ij: ij[0] < barloci)
    ln2 = order(ln2, last)[1:]
    ln = ln1 + ln2
    k = len(ln1)
    lineleninv = unifint(diff_lb, diff_ub, (0, k - 2))
    linelen = k - lineleninv
    givenl = ln[:linelen]
    reml = ln[linelen:]
    gi = fill(gi, linc, givenl)
    go = fill(gi, 3, reml)
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

IntegerSet = FrozenSet[Integer]

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

ONE = 1

THREE = 3

F = False

T = True

UNITY = (1, 1)

NEG_UNITY = (-1, -1)

UP_RIGHT = (-1, 1)

DOWN_LEFT = (1, -1)

def equality(
    a: Any,
    b: Any
) -> Boolean:
    """ equality """
    return a == b

def contained(
    value: Any,
    container: Container
) -> Boolean:
    """ element of """
    return value in container

def combine(
    a: Container,
    b: Container
) -> Container:
    """ union """
    return type(a)((*a, *b))

def intersection(
    a: FrozenSet,
    b: FrozenSet
) -> FrozenSet:
    """ returns the intersection of two containers """
    return a & b

def difference(
    a: Container,
    b: Container
) -> Container:
    """ difference """
    return type(a)(e for e in a if e not in b)

def greater(
    a: Integer,
    b: Integer
) -> Boolean:
    """ greater """
    return a > b

def maximum(
    container: IntegerSet
) -> Integer:
    """ maximum """
    return max(container, default=0)

def argmin(
    container: Container,
    compfunc: Callable
) -> Any:
    """ smallest item by custom order """
    return min(container, key=compfunc, default=None)

def initset(
    value: Any
) -> FrozenSet:
    """ initialize container """
    return frozenset({value})

def extract(
    container: Container,
    condition: Callable
) -> Any:
    """ first element of container that satisfies condition """
    return next(e for e in container if condition(e))

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

def manhattan(
    a: Patch,
    b: Patch
) -> Integer:
    """ closest manhattan distance between two patches """
    return min(abs(ai - bi) + abs(aj - bj) for ai, aj in toindices(a) for bi, bj in toindices(b))

def palette(
    element: Element
) -> IntegerSet:
    """ colors occurring in object or grid """
    if isinstance(element, tuple):
        return frozenset({v for r in element for v in r})
    return frozenset({v for v, _ in element})

def backdrop(
    patch: Patch
) -> Indices:
    """ indices in bounding box of patch """
    if len(patch) == 0:
        return frozenset({})
    indices = toindices(patch)
    si, sj = ulcorner(indices)
    ei, ej = lrcorner(patch)
    return frozenset((i, j) for i in range(si, ei + 1) for j in range(sj, ej + 1))

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

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_508bd3b6(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = objects(I, T, F, F)
    x1 = palette(I)
    x2 = compose(maximum, shape)
    x3 = lbind(apply, x2)
    x4 = lbind(colorfilter, x0)
    x5 = chain(maximum, x3, x4)
    x6 = matcher(x5, ONE)
    x7 = extract(x1, x6)
    x8 = lbind(ofcolor, I)
    x9 = compose(backdrop, x8)
    x10 = fork(equality, x8, x9)
    x11 = extract(x1, x10)
    x12 = ofcolor(I, x11)
    x13 = ofcolor(I, x7)
    x14 = rbind(manhattan, x12)
    x15 = compose(x14, initset)
    x16 = argmin(x13, x15)
    x17 = ulcorner(x13)
    x18 = contained(x17, x13)
    x19 = shoot(x16, UNITY)
    x20 = shoot(x16, NEG_UNITY)
    x21 = combine(x19, x20)
    x22 = shoot(x16, UP_RIGHT)
    x23 = shoot(x16, DOWN_LEFT)
    x24 = combine(x22, x23)
    x25 = branch(x18, x21, x24)
    x26 = asindices(I)
    x27 = outbox(x12)
    x28 = intersection(x26, x27)
    x29 = intersection(x28, x25)
    x30 = initset(x16)
    x31 = rbind(manhattan, x30)
    x32 = compose(x31, initset)
    x33 = argmin(x29, x32)
    x34 = height(x12)
    x35 = height(I)
    x36 = equality(x34, x35)
    x37 = leftmost(x13)
    x38 = leftmost(x12)
    x39 = greater(x37, x38)
    x40 = uppermost(x13)
    x41 = uppermost(x12)
    x42 = greater(x40, x41)
    x43 = lbind(shoot, x33)
    x44 = branch(x39, UNITY, NEG_UNITY)
    x45 = branch(x39, UP_RIGHT, DOWN_LEFT)
    x46 = branch(x42, UNITY, NEG_UNITY)
    x47 = branch(x42, DOWN_LEFT, UP_RIGHT)
    x48 = branch(x36, x44, x46)
    x49 = branch(x36, x45, x47)
    x50 = x43(x48)
    x51 = x43(x49)
    x52 = combine(x50, x51)
    x53 = difference(x52, x13)
    x54 = fill(I, THREE, x53)
    return x54


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_508bd3b6(inp)
        assert pred == _to_grid(expected), f"{name} failed"

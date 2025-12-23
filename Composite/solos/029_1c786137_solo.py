# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "1c786137"
SERIAL = "029"
URL    = "https://arcprize.org/play?task=1c786137"

# --- Code Golf Concepts ---
CONCEPTS = [
    "detect_enclosure",
    "crop",
]

# --- Example Grids ---
E1_IN = np.array([
    [3, 8, 8, 0, 3, 8, 8, 0, 8, 0, 3, 1, 1, 1, 8, 8, 0, 3, 8, 3, 8],
    [3, 3, 0, 0, 5, 3, 0, 3, 8, 0, 3, 3, 8, 1, 1, 8, 1, 3, 1, 8, 3],
    [1, 5, 1, 3, 1, 1, 8, 3, 0, 0, 3, 8, 3, 0, 1, 0, 8, 8, 5, 5, 0],
    [5, 3, 0, 8, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 0, 3, 0, 0, 3],
    [0, 1, 3, 3, 2, 0, 0, 8, 0, 3, 3, 3, 3, 2, 0, 0, 8, 0, 3, 3, 1],
    [8, 0, 0, 8, 2, 1, 0, 0, 0, 3, 0, 3, 1, 2, 0, 0, 0, 8, 0, 1, 0],
    [1, 1, 5, 0, 2, 3, 3, 0, 3, 3, 0, 8, 1, 2, 1, 0, 8, 3, 1, 0, 0],
    [0, 0, 8, 8, 2, 3, 3, 5, 1, 0, 3, 0, 0, 2, 1, 0, 5, 0, 3, 0, 1],
    [0, 1, 0, 0, 2, 5, 1, 3, 0, 1, 3, 1, 1, 2, 8, 8, 0, 5, 0, 3, 8],
    [8, 3, 3, 3, 2, 5, 0, 8, 0, 3, 0, 8, 8, 2, 3, 3, 0, 0, 3, 3, 8],
    [1, 1, 1, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 8, 1, 3, 0, 0],
    [3, 3, 3, 0, 8, 8, 0, 8, 3, 0, 8, 8, 3, 0, 3, 0, 8, 1, 0, 1, 0],
    [8, 0, 0, 3, 3, 0, 8, 3, 0, 3, 3, 0, 1, 3, 3, 1, 8, 0, 0, 3, 8],
    [5, 1, 5, 1, 8, 3, 5, 0, 8, 3, 3, 8, 1, 8, 0, 0, 0, 3, 0, 0, 5],
    [1, 3, 1, 0, 1, 3, 1, 0, 5, 0, 3, 3, 8, 0, 8, 3, 8, 8, 8, 0, 0],
    [5, 3, 3, 3, 3, 8, 8, 0, 1, 1, 0, 8, 5, 1, 3, 0, 0, 8, 3, 1, 0],
    [3, 1, 3, 3, 8, 0, 3, 8, 0, 3, 1, 8, 3, 1, 8, 1, 1, 3, 8, 1, 0],
    [0, 3, 8, 3, 3, 0, 1, 3, 0, 3, 8, 5, 3, 0, 3, 1, 0, 3, 0, 0, 8],
    [3, 8, 3, 0, 1, 3, 8, 0, 1, 3, 8, 1, 0, 1, 1, 8, 5, 8, 3, 1, 1],
    [1, 5, 1, 3, 3, 1, 5, 3, 3, 1, 1, 3, 5, 0, 8, 8, 1, 1, 8, 0, 8],
    [1, 3, 0, 1, 3, 3, 1, 0, 0, 1, 5, 8, 3, 5, 3, 8, 0, 3, 8, 3, 8],
    [3, 1, 3, 0, 8, 0, 8, 0, 0, 1, 3, 1, 1, 0, 8, 8, 5, 1, 0, 1, 8],
    [3, 3, 1, 0, 3, 1, 8, 8, 0, 0, 5, 1, 8, 8, 1, 3, 3, 5, 3, 5, 8],
], dtype=int)

E1_OUT = np.array([
    [0, 0, 8, 0, 3, 3, 3, 3],
    [1, 0, 0, 0, 3, 0, 3, 1],
    [3, 3, 0, 3, 3, 0, 8, 1],
    [3, 3, 5, 1, 0, 3, 0, 0],
    [5, 1, 3, 0, 1, 3, 1, 1],
    [5, 0, 8, 0, 3, 0, 8, 8],
], dtype=int)

E2_IN = np.array([
    [0, 6, 9, 6, 6, 0, 6, 3, 6, 9, 6, 6, 6, 9, 9, 0],
    [9, 9, 0, 6, 6, 0, 0, 9, 3, 6, 6, 6, 9, 9, 0, 6],
    [6, 0, 9, 0, 0, 6, 0, 6, 6, 0, 3, 0, 0, 6, 0, 0],
    [9, 6, 6, 9, 9, 9, 6, 3, 6, 9, 9, 6, 6, 3, 6, 6],
    [6, 6, 0, 0, 6, 6, 9, 0, 0, 3, 0, 0, 0, 0, 0, 9],
    [9, 9, 6, 0, 0, 9, 0, 0, 3, 9, 3, 0, 0, 0, 9, 0],
    [3, 6, 4, 4, 4, 4, 4, 6, 0, 0, 0, 9, 0, 0, 0, 9],
    [9, 0, 4, 3, 3, 0, 4, 0, 0, 6, 0, 0, 9, 6, 9, 3],
    [9, 0, 4, 9, 3, 9, 4, 9, 0, 0, 3, 9, 0, 0, 9, 3],
    [6, 9, 4, 6, 6, 0, 4, 3, 9, 6, 0, 6, 0, 9, 3, 0],
    [3, 3, 4, 9, 0, 0, 4, 9, 0, 6, 0, 0, 0, 6, 0, 0],
    [0, 0, 4, 6, 3, 9, 4, 6, 0, 9, 0, 9, 0, 0, 0, 0],
    [9, 9, 4, 4, 4, 4, 4, 9, 9, 0, 9, 9, 0, 0, 0, 6],
], dtype=int)

E2_OUT = np.array([
    [3, 3, 0],
    [9, 3, 9],
    [6, 6, 0],
    [9, 0, 0],
    [6, 3, 9],
], dtype=int)

E3_IN = np.array([
    [2, 5, 0, 0, 3, 0, 0, 2, 0, 0, 0, 0, 0, 0, 3, 5, 3, 5],
    [2, 0, 0, 2, 0, 2, 2, 2, 2, 2, 2, 5, 3, 0, 3, 2, 0, 5],
    [0, 5, 5, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 5, 0, 0],
    [2, 0, 2, 8, 0, 0, 5, 3, 3, 3, 2, 2, 5, 0, 8, 2, 5, 5],
    [5, 0, 3, 8, 3, 0, 0, 5, 5, 5, 5, 2, 0, 5, 8, 3, 3, 3],
    [0, 5, 5, 8, 3, 5, 0, 2, 0, 3, 0, 5, 3, 0, 8, 0, 2, 5],
    [5, 2, 2, 8, 3, 2, 5, 5, 0, 5, 3, 0, 5, 0, 8, 0, 0, 0],
    [0, 0, 0, 8, 5, 2, 5, 2, 5, 0, 2, 2, 2, 2, 8, 2, 0, 5],
    [5, 0, 5, 8, 0, 5, 2, 5, 0, 0, 0, 0, 3, 3, 8, 0, 0, 5],
    [3, 0, 0, 8, 2, 3, 2, 3, 0, 0, 5, 0, 5, 0, 8, 3, 2, 0],
    [3, 5, 0, 8, 3, 2, 5, 0, 5, 0, 0, 0, 5, 5, 8, 0, 0, 2],
    [3, 3, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 2, 0],
    [5, 0, 0, 3, 0, 3, 3, 5, 2, 5, 0, 0, 0, 0, 0, 5, 0, 0],
    [2, 5, 2, 5, 2, 2, 0, 0, 0, 5, 2, 0, 2, 0, 3, 0, 3, 0],
    [0, 2, 2, 2, 2, 0, 0, 2, 0, 2, 3, 3, 2, 0, 2, 5, 2, 5],
    [3, 0, 0, 0, 0, 5, 3, 0, 0, 0, 2, 2, 5, 0, 2, 3, 2, 0],
    [0, 0, 2, 5, 0, 5, 0, 3, 0, 0, 0, 0, 2, 3, 3, 5, 2, 3],
], dtype=int)

E3_OUT = np.array([
    [0, 0, 5, 3, 3, 3, 2, 2, 5, 0],
    [3, 0, 0, 5, 5, 5, 5, 2, 0, 5],
    [3, 5, 0, 2, 0, 3, 0, 5, 3, 0],
    [3, 2, 5, 5, 0, 5, 3, 0, 5, 0],
    [5, 2, 5, 2, 5, 0, 2, 2, 2, 2],
    [0, 5, 2, 5, 0, 0, 0, 0, 3, 3],
    [2, 3, 2, 3, 0, 0, 5, 0, 5, 0],
    [3, 2, 5, 0, 5, 0, 0, 0, 5, 5],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 0, 8, 1, 1, 8, 0, 0, 8, 0, 8, 0, 0, 0, 8],
    [0, 1, 0, 8, 8, 1, 0, 1, 1, 2, 8, 1, 1, 2, 0, 2],
    [0, 0, 8, 8, 1, 1, 8, 8, 1, 1, 8, 0, 8, 0, 0, 1],
    [1, 0, 1, 0, 8, 0, 1, 8, 1, 0, 1, 1, 8, 8, 8, 0],
    [8, 0, 8, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 2],
    [1, 0, 8, 3, 2, 0, 8, 1, 1, 1, 0, 1, 0, 3, 0, 0],
    [0, 8, 8, 3, 8, 1, 0, 8, 2, 8, 1, 2, 8, 3, 1, 8],
    [1, 0, 8, 3, 8, 2, 0, 2, 0, 1, 1, 8, 1, 3, 8, 8],
    [0, 8, 0, 3, 0, 1, 8, 8, 1, 1, 8, 1, 8, 3, 2, 1],
    [1, 0, 0, 3, 0, 1, 8, 8, 0, 8, 0, 2, 0, 3, 8, 1],
    [0, 8, 8, 3, 0, 8, 8, 2, 8, 8, 8, 8, 8, 3, 8, 8],
    [1, 1, 1, 3, 8, 0, 2, 0, 0, 0, 0, 8, 8, 3, 8, 0],
    [1, 8, 0, 3, 0, 2, 8, 8, 1, 2, 0, 0, 2, 3, 8, 1],
    [8, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 2],
    [8, 1, 0, 0, 0, 0, 8, 8, 0, 1, 2, 8, 8, 8, 1, 8],
    [8, 1, 0, 0, 1, 1, 8, 0, 1, 2, 8, 1, 0, 1, 2, 0],
    [8, 0, 8, 2, 8, 0, 8, 2, 0, 1, 8, 1, 8, 1, 8, 8],
], dtype=int)

T_OUT = np.array([
    [2, 0, 8, 1, 1, 1, 0, 1, 0],
    [8, 1, 0, 8, 2, 8, 1, 2, 8],
    [8, 2, 0, 2, 0, 1, 1, 8, 1],
    [0, 1, 8, 8, 1, 1, 8, 1, 8],
    [0, 1, 8, 8, 0, 8, 0, 2, 0],
    [0, 8, 8, 2, 8, 8, 8, 8, 8],
    [8, 0, 2, 0, 0, 0, 0, 8, 8],
    [0, 2, 8, 8, 1, 2, 0, 0, 2],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(g,L=len,E=enumerate):
 for C in set(sum(g,[])):
  P=[[x,y] for y,r in E(g) for x,c in E(r) if c==C]
  f=sum(P,[]);x=f[::2];y=f[1::2]
  X=g[min(y):max(y)]
  X=[r[min(x)+1:max(x)][:] for r in X]
  if X[0].count(C)==L(X[0]):
   return X[1:]
 return g


# --- Code Golf Solution (Compressed) ---
def q(g, w=9):
    return g * w and p(g, w - 1) + [(g := [r[~x::-1] for *r, in zip(*g) if w in r]) for x in [0] * 4 + [1] * 5][7] * ([] == g)


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, sample, uniform

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Numerical = Union[Integer, IntegerTuple]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

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

def difference(
    a: Container,
    b: Container
) -> Container:
    """ difference """
    return type(a)(e for e in a if e not in b)

def increment(
    x: Numerical
) -> Numerical:
    """ incrementing """
    return x + 1 if isinstance(x, int) else (x[0] + 1, x[1] + 1)

def toivec(
    i: Integer
) -> IntegerTuple:
    """ vector pointing vertically """
    return (i, 0)

def tojvec(
    j: Integer
) -> IntegerTuple:
    """ vector pointing horizontally """
    return (0, j)

def totuple(
    container: FrozenSet
) -> Tuple:
    """ conversion to tuple """
    return tuple(container)

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

def crop(
    grid: Grid,
    start: IntegerTuple,
    dims: IntegerTuple
) -> Grid:
    """ subgrid specified by start and dimension """
    return tuple(r[start[1]:start[1]+dims[1]] for r in grid[start[0]:start[0]+dims[0]])

def toindices(
    patch: Patch
) -> Indices:
    """ indices of object cells """
    if len(patch) == 0:
        return frozenset()
    if isinstance(next(iter(patch))[1], tuple):
        return frozenset(index for value, index in patch)
    return patch

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

def generate_1c786137(diff_lb: float, diff_ub: float) -> dict:
    dim_bounds = (3, 30)
    num_cols_card_bounds = (1, 8)
    colopts = interval(1, 10, 1)
    h = unifint(diff_lb, diff_ub, dim_bounds)
    w = unifint(diff_lb, diff_ub, dim_bounds)
    noise_card_bounds = (0, h * w)
    c = canvas(0, (h, w))
    inds = totuple(asindices(c))
    num_noise = unifint(diff_lb, diff_ub, noise_card_bounds)
    num_cols = unifint(diff_lb, diff_ub, num_cols_card_bounds)
    noiseinds = sample(inds, num_noise)
    colset = sample(colopts, num_cols)
    trgcol = choice(difference(colopts, colset))
    noise = frozenset((choice(colset), ij) for ij in noiseinds)
    gi = paint(c, noise)
    boxhrng = (3, max(3, h//2))
    boxwrng = (3, max(3, w//2))
    boxh = unifint(diff_lb, diff_ub, boxhrng)
    boxw = unifint(diff_lb, diff_ub, boxwrng)
    boxi = choice(interval(0, h - boxh + 1, 1))
    boxj = choice(interval(0, w - boxw + 1, 1))
    loc = (boxi, boxj)
    llc = add(loc, toivec(boxh - 1))
    urc = add(loc, tojvec(boxw - 1))
    lrc = add(loc, (boxh - 1, boxw - 1))
    l1 = connect(loc, llc)
    l2 = connect(loc, urc)
    l3 = connect(urc, lrc)
    l4 = connect(llc, lrc)
    l = l1 | l2 | l3 | l4
    gi = fill(gi, trgcol, l)
    go = crop(gi, increment(loc), (boxh - 2, boxw - 2))
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

IntegerSet = FrozenSet[Integer]

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

ONE = 1

SEVEN = 7

F = False

T = True

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

def greater(
    a: Integer,
    b: Integer
) -> Boolean:
    """ greater """
    return a > b

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

def extract(
    container: Container,
    condition: Callable
) -> Any:
    """ first element of container that satisfies condition """
    return next(e for e in container if condition(e))

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

def fork(
    outer: Callable,
    a: Callable,
    b: Callable
) -> Callable:
    """ creates a wrapper function """
    return lambda x: outer(a(x), b(x))

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

def color(
    obj: Object
) -> Integer:
    """ color of object """
    return next(iter(obj))[0]

def subgrid(
    patch: Patch,
    grid: Grid
) -> Grid:
    """ smallest subgrid containing object """
    return crop(grid, ulcorner(patch), shape(patch))

def trim(
    grid: Grid
) -> Grid:
    """ trim border of grid """
    return tuple(r[1:-1] for r in grid[1:-1])

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

def verify_1c786137(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = objects(I, T, F, F)
    x1 = lbind(colorfilter, x0)
    x2 = compose(size, x1)
    x3 = matcher(x2, ONE)
    x4 = palette(I)
    x5 = sfilter(x4, x3)
    x6 = fork(equality, toindices, box)
    x7 = rbind(contained, x5)
    x8 = compose(x7, color)
    x9 = sfilter(x0, x8)
    x10 = rbind(greater, SEVEN)
    x11 = compose(x10, size)
    x12 = sfilter(x9, x11)
    x13 = extract(x12, x6)
    x14 = subgrid(x13, I)
    x15 = trim(x14)
    return x15


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_1c786137(inp)
        assert pred == _to_grid(expected), f"{name} failed"

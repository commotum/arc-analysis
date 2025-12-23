# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "feca6190"
SERIAL = "398"
URL    = "https://arcprize.org/play?task=feca6190"

# --- Code Golf Concepts ---
CONCEPTS = [
    "pattern_expansion",
    "image_expansion",
    "draw_line_from_point",
    "diagonals",
]

# --- Example Grids ---
E1_IN = np.array([
    [1, 0, 7, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 7],
    [0, 0, 0, 0, 0, 0, 1, 0, 7, 0],
    [0, 0, 0, 0, 0, 1, 0, 7, 0, 0],
    [0, 0, 0, 0, 1, 0, 7, 0, 0, 0],
    [0, 0, 0, 1, 0, 7, 0, 0, 0, 0],
    [0, 0, 1, 0, 7, 0, 0, 0, 0, 0],
    [0, 1, 0, 7, 0, 0, 0, 0, 0, 0],
    [1, 0, 7, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E2_IN = np.array([
    [0, 0, 2, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 2],
    [0, 0, 0, 2, 0],
    [0, 0, 2, 0, 0],
], dtype=int)

E3_IN = np.array([
    [4, 0, 6, 0, 8],
], dtype=int)

E3_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 6],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 6, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 6, 0, 8],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 6, 0, 8, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 6, 0, 8, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 4, 0, 6, 0, 8, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 4, 0, 6, 0, 8, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 4, 0, 6, 0, 8, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 4, 0, 6, 0, 8, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 4, 0, 6, 0, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 4, 0, 6, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 4, 0, 6, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 0, 6, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E4_IN = np.array([
    [0, 9, 0, 8, 4],
], dtype=int)

E4_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 8],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 8, 4],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 8, 4, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 8, 4, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 8, 4, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 9, 0, 8, 4, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 9, 0, 8, 4, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 9, 0, 8, 4, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 9, 0, 8, 4, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 9, 0, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 9, 0, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 9, 0, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E5_IN = np.array([
    [0, 4, 0, 0, 0],
], dtype=int)

E5_OUT = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 4],
    [0, 0, 0, 4, 0],
    [0, 0, 4, 0, 0],
    [0, 4, 0, 0, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 6, 7, 8, 9],
], dtype=int)

T_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 7],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 7, 8],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 7, 8, 9],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(g,L=len,R=range):
 s=R(L([x for x in set(g[0])if x>0])*5)
 X=[[0 for x in s]for y in s]
 g=g[0]
 T=0
 for r in s:
  for c in R(5):
   try:X[-(r+1)][c+T]=g[c]
   except:pass
  T+=1
 return X


# --- Code Golf Solution (Compressed) ---
def q(g):
    g, = g
    r = ~g.count(0) % 6 * 5 * [0]
    return [(r := (r[1:] + [c])) for c in g + r[5:]]


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import sample, uniform

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

UP_RIGHT = (-1, 1)

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

def generate_feca6190(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    w = unifint(diff_lb, diff_ub, (2, 6))
    bgc = 0
    remcols = remove(bgc, cols)
    ncols = unifint(diff_lb, diff_ub, (1, min(w, 5)))
    ccols = sample(remcols, ncols)
    cands = interval(0, w, 1)
    locs = sample(cands, ncols)
    gi = canvas(bgc, (1, w))
    go = canvas(bgc, (w*ncols, w*ncols))
    for col, j in zip(ccols, locs):
        gi = fill(gi, col, {(0, j)})
        go = fill(go, col, shoot((w*ncols-1, j), UP_RIGHT))
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Numerical = Union[Integer, IntegerTuple]

Piece = Union[Grid, Patch]

ContainerContainer = Container[Container]

ZERO = 0

UNITY = (1, 1)

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

def flip(
    b: Boolean
) -> Boolean:
    """ logical not """
    return not b

def size(
    container: Container
) -> Integer:
    """ cardinality """
    return len(container)

def merge(
    containers: ContainerContainer
) -> Container:
    """ merging """
    return type(containers)(e for c in containers for e in c)

def decrement(
    x: Numerical
) -> Numerical:
    """ decrementing """
    return x - 1 if isinstance(x, int) else (x[0] - 1, x[1] - 1)

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

def astuple(
    a: Integer,
    b: Integer
) -> IntegerTuple:
    """ constructs a tuple """
    return (a, b)

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

def mapply(
    function: Callable,
    container: ContainerContainer
) -> FrozenSet:
    """ apply and merge """
    return merge(apply(function, container))

def width(
    piece: Piece
) -> Integer:
    """ width of grid or patch """
    if len(piece) == 0:
        return 0
    if isinstance(piece, tuple):
        return len(piece[0])
    return rightmost(piece) - leftmost(piece) + 1

def recolor(
    value: Integer,
    patch: Patch
) -> Object:
    """ recolor patch """
    return frozenset((value, index) for index in toindices(patch))

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

def verify_feca6190(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = asobject(I)
    x1 = matcher(first, ZERO)
    x2 = compose(flip, x1)
    x3 = sfilter(x0, x2)
    x4 = size(x3)
    x5 = width(I)
    x6 = multiply(x5, x4)
    x7 = multiply(UNITY, x6)
    x8 = canvas(ZERO, x7)
    x9 = multiply(x5, x4)
    x10 = decrement(x9)
    x11 = lbind(astuple, x10)
    x12 = rbind(shoot, UP_RIGHT)
    x13 = compose(last, last)
    x14 = chain(x12, x11, x13)
    x15 = fork(recolor, first, x14)
    x16 = mapply(x15, x3)
    x17 = paint(x8, x16)
    return x17


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("E4", E4_IN, E4_OUT),
        ("E5", E5_IN, E5_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_feca6190(inp)
        assert pred == _to_grid(expected), f"{name} failed"

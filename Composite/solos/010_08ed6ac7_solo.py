# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "08ed6ac7"
SERIAL = "010"
URL    = "https://arcprize.org/play?task=08ed6ac7"

# --- Code Golf Concepts ---
CONCEPTS = [
    "measure_length",
    "order_numbers",
    "associate_colors_to_ranks",
    "recoloring",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 0, 0, 0, 5, 0, 0, 0],
    [0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 5, 0, 5, 0, 5, 0, 0, 0],
    [0, 5, 0, 5, 0, 5, 0, 0, 0],
    [0, 5, 0, 5, 0, 5, 0, 0, 0],
    [0, 5, 0, 5, 0, 5, 0, 5, 0],
    [0, 5, 0, 5, 0, 5, 0, 5, 0],
    [0, 5, 0, 5, 0, 5, 0, 5, 0],
], dtype=int)

E1_OUT = np.array([
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 2, 0, 0, 0, 1, 0, 0, 0],
    [0, 2, 0, 0, 0, 1, 0, 0, 0],
    [0, 2, 0, 3, 0, 1, 0, 0, 0],
    [0, 2, 0, 3, 0, 1, 0, 0, 0],
    [0, 2, 0, 3, 0, 1, 0, 0, 0],
    [0, 2, 0, 3, 0, 1, 0, 4, 0],
    [0, 2, 0, 3, 0, 1, 0, 4, 0],
    [0, 2, 0, 3, 0, 1, 0, 4, 0],
], dtype=int)

E2_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 5, 0],
    [0, 0, 0, 0, 0, 0, 0, 5, 0],
    [0, 0, 0, 0, 0, 0, 0, 5, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0],
    [0, 0, 0, 5, 0, 5, 0, 5, 0],
    [0, 0, 0, 5, 0, 5, 0, 5, 0],
    [0, 5, 0, 5, 0, 5, 0, 5, 0],
    [0, 5, 0, 5, 0, 5, 0, 5, 0],
], dtype=int)

E2_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 2, 0, 0, 0, 1, 0],
    [0, 0, 0, 2, 0, 3, 0, 1, 0],
    [0, 0, 0, 2, 0, 3, 0, 1, 0],
    [0, 4, 0, 2, 0, 3, 0, 1, 0],
    [0, 4, 0, 2, 0, 3, 0, 1, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 5, 0, 0, 0, 0, 0, 0, 0],
    [0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 5, 0, 0, 0, 5, 0, 5, 0],
    [0, 5, 0, 0, 0, 5, 0, 5, 0],
    [0, 5, 0, 5, 0, 5, 0, 5, 0],
    [0, 5, 0, 5, 0, 5, 0, 5, 0],
    [0, 5, 0, 5, 0, 5, 0, 5, 0],
], dtype=int)

T_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 2, 0, 0, 0],
    [0, 1, 0, 0, 0, 2, 0, 0, 0],
    [0, 1, 0, 0, 0, 2, 0, 3, 0],
    [0, 1, 0, 0, 0, 2, 0, 3, 0],
    [0, 1, 0, 4, 0, 2, 0, 3, 0],
    [0, 1, 0, 4, 0, 2, 0, 3, 0],
    [0, 1, 0, 4, 0, 2, 0, 3, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
 A={}
 for c in j:
  for E,k in enumerate(c):
   if k==5:c[E]=A.setdefault(E,len(A)+1)
 return j


# --- Code Golf Solution (Compressed) ---
def q(g):
    return [(g := [(x * -5 or y * sum(r)) % 6 for x, y in zip(g, r)]) for r in g]


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, sample, shuffle, uniform

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

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

def generate_08ed6ac7(diff_lb: float, diff_ub: float) -> dict:
    colopts = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    bgc = choice(difference(colopts, (1, 2, 3, 4)))
    remcols = remove(bgc, colopts)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    barrange = (4, w)
    locopts = interval(0, w, 1)
    nbars = unifint(diff_lb, diff_ub, barrange)
    barlocs = sample(locopts, nbars)
    barhopts = interval(0, h, 1)
    barhs = sample(barhopts, 4)
    barcols = [choice(remcols) for j in range(nbars)]
    barhsfx = [choice(barhs) for j in range(nbars - 4)] + list(barhs)
    shuffle(barhsfx)
    ordered = sorted(barhs)
    colord = interval(1, 5, 1)
    for col, (loci, locj) in zip(barcols, list(zip(barhsfx, barlocs))):
        bar = connect((loci, locj), (h - 1, locj))
        gi = fill(gi, col, bar)
        go = fill(go, colord[ordered.index(loci)], bar)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Numerical = Union[Integer, IntegerTuple]

Piece = Union[Grid, Patch]

TupleTuple = Tuple[Tuple]

ONE = 1

def identity(
    x: Any
) -> Any:
    """ identity function """
    return x

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

def combine(
    a: Container,
    b: Container
) -> Container:
    """ union """
    return type(a)((*a, *b))

def dedupe(
    iterable: Tuple
) -> Tuple:
    """ remove duplicates """
    return tuple(e for i, e in enumerate(iterable) if iterable.index(e) == i)

def order(
    container: Container,
    compfunc: Callable
) -> Tuple:
    """ order container by custom key """
    return tuple(sorted(container, key=compfunc))

def repeat(
    item: Any,
    num: Integer
) -> Tuple:
    """ repetition of item within vector """
    return tuple(item for i in range(num))

def size(
    container: Container
) -> Integer:
    """ cardinality """
    return len(container)

def mostcommon(
    container: Container
) -> Any:
    """ most common item """
    return max(set(container), key=container.count)

def increment(
    x: Numerical
) -> Numerical:
    """ incrementing """
    return x + 1 if isinstance(x, int) else (x[0] + 1, x[1] + 1)

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

def height(
    piece: Piece
) -> Integer:
    """ height of grid or patch """
    if len(piece) == 0:
        return 0
    if isinstance(piece, tuple):
        return len(piece)
    return lowermost(piece) - uppermost(piece) + 1

def ulcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of upper left corner """
    return tuple(map(min, zip(*toindices(patch))))

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

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_08ed6ac7(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = first(I)
    x1 = mostcommon(x0)
    x2 = dmirror(I)
    x3 = matcher(identity, x1)
    x4 = rbind(sfilter, x3)
    x5 = compose(size, x4)
    x6 = apply(x5, x2)
    x7 = dedupe(x6)
    x8 = order(x7, identity)
    x9 = size(x8)
    x10 = increment(x9)
    x11 = increment(x10)
    x12 = interval(ONE, x11, ONE)
    x13 = pair(x8, x12)
    x14 = height(I)
    x15 = astuple(x14, x1)
    x16 = repeat(x15, ONE)
    x17 = combine(x16, x13)
    x18 = lbind(extract, x17)
    x19 = lbind(matcher, first)
    x20 = chain(last, x18, x19)
    x21 = compose(x20, x5)
    x22 = fork(subtract, height, x5)
    x23 = fork(repeat, x21, x22)
    x24 = lbind(repeat, x1)
    x25 = compose(x24, x5)
    x26 = fork(combine, x25, x23)
    x27 = apply(x26, x2)
    x28 = dmirror(x27)
    return x28


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_08ed6ac7(inp)
        assert pred == _to_grid(expected), f"{name} failed"

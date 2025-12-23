# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "9dfd6313"
SERIAL = "241"
URL    = "https://arcprize.org/play?task=9dfd6313"

# --- Code Golf Concepts ---
CONCEPTS = [
    "image_reflection",
    "diagonal_symmetry",
]

# --- Example Grids ---
E1_IN = np.array([
    [5, 0, 0],
    [3, 5, 0],
    [0, 0, 5],
], dtype=int)

E1_OUT = np.array([
    [5, 3, 0],
    [0, 5, 0],
    [0, 0, 5],
], dtype=int)

E2_IN = np.array([
    [5, 0, 0, 0],
    [0, 5, 0, 0],
    [6, 0, 5, 0],
    [6, 0, 4, 5],
], dtype=int)

E2_OUT = np.array([
    [5, 0, 6, 6],
    [0, 5, 0, 0],
    [0, 0, 5, 4],
    [0, 0, 0, 5],
], dtype=int)

E3_IN = np.array([
    [5, 0, 0, 0, 0],
    [0, 5, 0, 0, 0],
    [8, 8, 5, 0, 0],
    [0, 2, 0, 5, 0],
    [0, 2, 0, 1, 5],
], dtype=int)

E3_OUT = np.array([
    [5, 0, 8, 0, 0],
    [0, 5, 8, 2, 2],
    [0, 0, 5, 0, 0],
    [0, 0, 0, 5, 1],
    [0, 0, 0, 0, 5],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [5, 0, 0, 0, 0, 0],
    [0, 5, 0, 0, 0, 0],
    [3, 3, 5, 0, 0, 0],
    [0, 0, 0, 5, 0, 0],
    [2, 0, 8, 8, 5, 0],
    [2, 0, 6, 0, 0, 5],
], dtype=int)

T_OUT = np.array([
    [5, 0, 3, 0, 2, 2],
    [0, 5, 3, 0, 0, 0],
    [0, 0, 5, 0, 8, 6],
    [0, 0, 0, 5, 8, 0],
    [0, 0, 0, 0, 5, 0],
    [0, 0, 0, 0, 0, 5],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
    return [*map(list, zip(*j))]


# --- Code Golf Solution (Compressed) ---
def q(m):
    return [*zip(*m)]


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

def sfilter(
    container: Container,
    condition: Callable
) -> Container:
    """ keep elements in container that satisfy condition """
    return type(container)(e for e in container if condition(e))

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

def generate_9dfd6313(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    dh = unifint(diff_lb, diff_ub, (1, 14))
    d = 2 * dh + 1
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    linc = choice(remcols)
    remcols = remove(linc, remcols)
    gi = canvas(bgc, (d, d))
    inds = asindices(gi)
    lni = randint(1, 4)
    if lni == 1:
        ln = connect((dh, 0), (dh, d - 1))
        mirrf = hmirror
        cands = sfilter(inds, lambda ij: ij[0] > dh)
    elif lni == 2:
        ln = connect((0, dh), (d - 1, dh))
        mirrf = vmirror
        cands = sfilter(inds, lambda ij: ij[1] > dh)
    elif lni == 3:
        ln = connect((0, 0), (d - 1, d - 1))
        mirrf = dmirror
        cands = sfilter(inds, lambda ij: ij[0] > ij[1])
    elif lni == 4:
        ln = connect((d - 1, 0), (0, d - 1))
        mirrf = cmirror
        cands = sfilter(inds, lambda ij: (ij[0] + ij[1]) > d)
    gi = fill(gi, linc, ln)
    mp = (d * (d - 1)) // 2
    numcols = unifint(diff_lb, diff_ub, (1, min(7, mp)))
    colsch = sample(remcols, numcols)
    numpix = unifint(diff_lb, diff_ub, (1, len(cands)))
    pixs = sample(totuple(cands), numpix)
    for pix in pixs:
        gi = fill(gi, choice(colsch), {pix})
    go = mirrf(gi)
    if choice((True, False)):
        gi, go = go, gi
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Numerical = Union[Integer, IntegerTuple]

IntegerSet = FrozenSet[Integer]

Element = Union[Object, Grid]

ONE = 1

ORIGIN = (0, 0)

def halve(
    n: Numerical
) -> Numerical:
    """ scaling by one half """
    return n // 2 if isinstance(n, int) else (n[0] // 2, n[1] // 2)

def equality(
    a: Any,
    b: Any
) -> Boolean:
    """ equality """
    return a == b

def combine(
    a: Container,
    b: Container
) -> Container:
    """ union """
    return type(a)((*a, *b))

def size(
    container: Container
) -> Integer:
    """ cardinality """
    return len(container)

def both(
    a: Boolean,
    b: Boolean
) -> Boolean:
    """ logical and """
    return a and b

def decrement(
    x: Numerical
) -> Numerical:
    """ decrementing """
    return x - 1 if isinstance(x, int) else (x[0] - 1, x[1] - 1)

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

def colorcount(
    element: Element,
    value: Integer
) -> Integer:
    """ number of cells with color """
    if isinstance(element, tuple):
        return sum(row.count(value) for row in element)
    return sum(v == value for v, _ in element)

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

def numcolors(
    element: Element
) -> IntegerSet:
    """ number of colors occurring in object or grid """
    return len(palette(element))

def color(
    obj: Object
) -> Integer:
    """ color of object """
    return next(iter(obj))[0]

def toobject(
    patch: Patch,
    grid: Grid
) -> Object:
    """ object from patch and grid """
    h, w = len(grid), len(grid[0])
    return frozenset((grid[i][j], (i, j)) for i, j in toindices(patch) if 0 <= i < h and 0 <= j < w)

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_9dfd6313(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = shape(I)
    x1 = decrement(x0)
    x2 = connect(ORIGIN, x1)
    x3 = height(I)
    x4 = decrement(x3)
    x5 = toivec(x4)
    x6 = width(I)
    x7 = decrement(x6)
    x8 = tojvec(x7)
    x9 = connect(x5, x8)
    x10 = height(I)
    x11 = halve(x10)
    x12 = toivec(x11)
    x13 = width(I)
    x14 = decrement(x13)
    x15 = astuple(x11, x14)
    x16 = connect(x12, x15)
    x17 = width(I)
    x18 = halve(x17)
    x19 = tojvec(x18)
    x20 = height(I)
    x21 = decrement(x20)
    x22 = astuple(x21, x18)
    x23 = connect(x19, x22)
    x24 = astuple(x2, dmirror)
    x25 = astuple(x9, cmirror)
    x26 = astuple(x24, x25)
    x27 = astuple(x23, vmirror)
    x28 = astuple(x16, hmirror)
    x29 = astuple(x27, x28)
    x30 = combine(x26, x29)
    x31 = lbind(colorcount, I)
    x32 = rbind(toobject, I)
    x33 = compose(x32, first)
    x34 = chain(x31, color, x33)
    x35 = compose(size, first)
    x36 = fork(equality, x34, x35)
    x37 = rbind(toobject, I)
    x38 = chain(numcolors, x37, first)
    x39 = matcher(x38, ONE)
    x40 = fork(both, x39, x36)
    x41 = extract(x30, x40)
    x42 = last(x41)
    x43 = x42(I)
    return x43


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_9dfd6313(inp)
        assert pred == _to_grid(expected), f"{name} failed"

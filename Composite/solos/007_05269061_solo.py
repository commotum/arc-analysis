# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "05269061"
SERIAL = "007"
URL    = "https://arcprize.org/play?task=05269061"

# --- Code Golf Concepts ---
CONCEPTS = [
    "image_filling",
    "pattern_expansion",
    "diagonals",
]

# --- Example Grids ---
E1_IN = np.array([
    [2, 8, 3, 0, 0, 0, 0],
    [8, 3, 0, 0, 0, 0, 0],
    [3, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [2, 8, 3, 2, 8, 3, 2],
    [8, 3, 2, 8, 3, 2, 8],
    [3, 2, 8, 3, 2, 8, 3],
    [2, 8, 3, 2, 8, 3, 2],
    [8, 3, 2, 8, 3, 2, 8],
    [3, 2, 8, 3, 2, 8, 3],
    [2, 8, 3, 2, 8, 3, 2],
], dtype=int)

E2_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 2],
    [0, 0, 0, 0, 1, 2, 4],
    [0, 0, 0, 1, 2, 4, 0],
    [0, 0, 1, 2, 4, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [2, 4, 1, 2, 4, 1, 2],
    [4, 1, 2, 4, 1, 2, 4],
    [1, 2, 4, 1, 2, 4, 1],
    [2, 4, 1, 2, 4, 1, 2],
    [4, 1, 2, 4, 1, 2, 4],
    [1, 2, 4, 1, 2, 4, 1],
    [2, 4, 1, 2, 4, 1, 2],
], dtype=int)

E3_IN = np.array([
    [0, 0, 0, 0, 8, 3, 0],
    [0, 0, 0, 8, 3, 0, 0],
    [0, 0, 8, 3, 0, 0, 0],
    [0, 8, 3, 0, 0, 0, 4],
    [8, 3, 0, 0, 0, 4, 0],
    [3, 0, 0, 0, 4, 0, 0],
    [0, 0, 0, 4, 0, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [4, 8, 3, 4, 8, 3, 4],
    [8, 3, 4, 8, 3, 4, 8],
    [3, 4, 8, 3, 4, 8, 3],
    [4, 8, 3, 4, 8, 3, 4],
    [8, 3, 4, 8, 3, 4, 8],
    [3, 4, 8, 3, 4, 8, 3],
    [4, 8, 3, 4, 8, 3, 4],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 1, 0, 0, 0, 0, 2],
    [1, 0, 0, 0, 0, 2, 0],
    [0, 0, 0, 0, 2, 0, 0],
    [0, 0, 0, 2, 0, 0, 0],
    [0, 0, 2, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 4],
    [2, 0, 0, 0, 0, 4, 0],
], dtype=int)

T_OUT = np.array([
    [2, 1, 4, 2, 1, 4, 2],
    [1, 4, 2, 1, 4, 2, 1],
    [4, 2, 1, 4, 2, 1, 4],
    [2, 1, 4, 2, 1, 4, 2],
    [1, 4, 2, 1, 4, 2, 1],
    [4, 2, 1, 4, 2, 1, 4],
    [2, 1, 4, 2, 1, 4, 2],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(g):R=range;L=len;d={(i+j)%3:c for i in R(L(g))for j in R(L(g[0]))for c in[g[i][j]]if c};return[[d.get((i+j)%3,0)for j in R(L(g[0]))]for i in R(L(g))]


# --- Code Golf Solution (Compressed) ---
def q(g):
    return eval(f'({'max(sum(g:=g[1:3]+g,[0])[::3]),' * 7}),' * 7)


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, uniform

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

Piece = Union[Grid, Patch]

UP_RIGHT = (-1, 1)

def toivec(
    i: Integer
) -> IntegerTuple:
    """ vector pointing vertically """
    return (i, 0)

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

def hfrontier(
    location: IntegerTuple
) -> Indices:
    """ horizontal frontier """
    return frozenset((location[0], j) for j in range(30))

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

def generate_05269061(diff_lb: float, diff_ub: float) -> dict:
    dim_bounds = (2, 30)
    colopts = interval(1, 10, 1)
    d = unifint(diff_lb, diff_ub, dim_bounds)
    go = canvas(0, (d, d))
    gi = canvas(0, (d, d))
    if choice((True, False)):
        period_bounds = (2, min(2*d-2, 9))
        num = unifint(diff_lb, diff_ub, period_bounds)
        cols = tuple(choice(colopts) for k in range(num))
        keeps = [choice(interval(j, 2*d-1, num)) for j in range(num)]
        for k, col in enumerate((cols * 30)[:2*d-1]):
            lin = shoot(toivec(k), UP_RIGHT)
            go = fill(go, col, lin)
            if keeps[k % num] == k:
                gi = fill(gi, col, lin)
    else:
        period_bounds = (2, min(d, 9))
        num = unifint(diff_lb, diff_ub, period_bounds)
        cols = tuple(choice(colopts) for k in range(num))
        keeps = [choice(interval(j, d, num)) for j in range(num)]
        for k, col in enumerate((cols * 30)[:d]):
            lin = hfrontier(toivec(k))
            go = fill(go, col, lin)
            if keeps[k % num] == k:
                gi = fill(gi, col, lin)
    if choice((True, False)):
        gi = vmirror(gi)
        go = vmirror(go)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Numerical = Union[Integer, IntegerTuple]

IntegerSet = FrozenSet[Integer]

Element = Union[Object, Grid]

TupleTuple = Tuple[Tuple]

ContainerContainer = Container[Container]

ZERO = 0

ONE = 1

DOWN = (1, 0)

RIGHT = (0, 1)

NEG_UNITY = (-1, -1)

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

def double(
    n: Numerical
) -> Numerical:
    """ scaling by two """
    return n * 2 if isinstance(n, int) else (n[0] * 2, n[1] * 2)

def flip(
    b: Boolean
) -> Boolean:
    """ logical not """
    return not b

def combine(
    a: Container,
    b: Container
) -> Container:
    """ union """
    return type(a)((*a, *b))

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

def valmax(
    container: Container,
    compfunc: Callable
) -> Integer:
    """ maximum by custom function """
    return compfunc(max(container, key=compfunc, default=0))

def increment(
    x: Numerical
) -> Numerical:
    """ incrementing """
    return x + 1 if isinstance(x, int) else (x[0] + 1, x[1] + 1)

def decrement(
    x: Numerical
) -> Numerical:
    """ decrementing """
    return x - 1 if isinstance(x, int) else (x[0] - 1, x[1] - 1)

def tojvec(
    j: Integer
) -> IntegerTuple:
    """ vector pointing horizontally """
    return (0, j)

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

def pair(
    a: Tuple,
    b: Tuple
) -> TupleTuple:
    """ zipping of two tuples """
    return tuple(zip(a, b))

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

def shape(
    piece: Piece
) -> IntegerTuple:
    """ height and width of grid or patch """
    return (height(piece), width(piece))

def recolor(
    value: Integer,
    patch: Patch
) -> Object:
    """ recolor patch """
    return frozenset((value, index) for index in toindices(patch))

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

def toobject(
    patch: Patch,
    grid: Grid
) -> Object:
    """ object from patch and grid """
    h, w = len(grid), len(grid[0])
    return frozenset((grid[i][j], (i, j)) for i, j in toindices(patch) if 0 <= i < h and 0 <= j < w)

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

def verify_05269061(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = shape(I)
    x1 = maximum(x0)
    x2 = interval(ZERO, x1, ONE)
    x3 = interval(ONE, x1, ONE)
    x4 = rbind(toobject, I)
    x5 = rbind(shoot, RIGHT)
    x6 = chain(x4, x5, toivec)
    x7 = rbind(shoot, DOWN)
    x8 = chain(x4, x7, tojvec)
    x9 = apply(x6, x2)
    x10 = apply(x8, x2)
    x11 = rbind(shoot, UP_RIGHT)
    x12 = chain(x4, x11, toivec)
    x13 = rbind(shoot, UP_RIGHT)
    x14 = decrement(x1)
    x15 = lbind(astuple, x14)
    x16 = chain(x4, x13, x15)
    x17 = apply(x12, x2)
    x18 = apply(x16, x3)
    x19 = combine(x17, x18)
    x20 = rbind(shoot, NEG_UNITY)
    x21 = decrement(x1)
    x22 = lbind(astuple, x21)
    x23 = chain(x4, x20, x22)
    x24 = rbind(shoot, NEG_UNITY)
    x25 = decrement(x1)
    x26 = rbind(astuple, x25)
    x27 = lbind(subtract, x25)
    x28 = compose(x26, x27)
    x29 = chain(x4, x24, x28)
    x30 = apply(x23, x2)
    x31 = apply(x29, x3)
    x32 = combine(x30, x31)
    x33 = rbind(valmax, numcolors)
    x34 = matcher(x33, ONE)
    x35 = x34(x9)
    x36 = x34(x10)
    x37 = x34(x19)
    x38 = branch(x37, x19, x32)
    x39 = branch(x36, x10, x38)
    x40 = branch(x35, x9, x39)
    x41 = apply(mostcolor, x40)
    x42 = matcher(identity, ZERO)
    x43 = compose(flip, x42)
    x44 = sfilter(x41, x43)
    x45 = size(x44)
    x46 = double(x1)
    x47 = divide(x46, x45)
    x48 = increment(x47)
    x49 = interval(ZERO, x48, ONE)
    x50 = matcher(first, ZERO)
    x51 = compose(flip, x50)
    x52 = fork(recolor, first, last)
    x53 = size(x40)
    x54 = interval(ZERO, x53, ONE)
    x55 = rbind(compose, first)
    x56 = lbind(rbind, greater)
    x57 = chain(x55, x56, decrement)
    x58 = lbind(apply, last)
    x59 = lbind(chain, x58)
    x60 = rbind(x59, x57)
    x61 = lbind(lbind, sfilter)
    x62 = lbind(pair, x54)
    x63 = chain(x60, x61, x62)
    x64 = x63(x40)
    x65 = x63(x41)
    x66 = rbind(multiply, x45)
    x67 = compose(x64, x66)
    x68 = rbind(multiply, x45)
    x69 = compose(x65, x68)
    x70 = lbind(mapply, x52)
    x71 = rbind(sfilter, x51)
    x72 = lbind(pair, x41)
    x73 = compose(x72, x67)
    x74 = chain(x70, x71, x73)
    x75 = lbind(mapply, x52)
    x76 = rbind(sfilter, x51)
    x77 = rbind(pair, x40)
    x78 = compose(x77, x69)
    x79 = chain(x75, x76, x78)
    x80 = fork(combine, x74, x79)
    x81 = mapply(x80, x49)
    x82 = paint(I, x81)
    return x82


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_05269061(inp)
        assert pred == _to_grid(expected), f"{name} failed"

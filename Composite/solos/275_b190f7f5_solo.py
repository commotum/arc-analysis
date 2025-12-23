# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "b190f7f5"
SERIAL = "275"
URL    = "https://arcprize.org/play?task=b190f7f5"

# --- Code Golf Concepts ---
CONCEPTS = [
    "separate_images",
    "image_expasion",
    "color_palette",
    "image_resizing",
    "replace_pattern",
]

# --- Example Grids ---
E1_IN = np.array([
    [2, 0, 4, 0, 8, 0],
    [0, 3, 0, 8, 8, 8],
    [0, 0, 0, 0, 8, 0],
], dtype=int)

E1_OUT = np.array([
    [0, 2, 0, 0, 0, 0, 0, 4, 0],
    [2, 2, 2, 0, 0, 0, 4, 4, 4],
    [0, 2, 0, 0, 0, 0, 0, 4, 0],
    [0, 0, 0, 0, 3, 0, 0, 0, 0],
    [0, 0, 0, 3, 3, 3, 0, 0, 0],
    [0, 0, 0, 0, 3, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E2_IN = np.array([
    [0, 0, 8, 0, 4, 0],
    [8, 0, 0, 1, 2, 4],
    [8, 8, 0, 0, 1, 0],
], dtype=int)

E2_OUT = np.array([
    [0, 0, 0, 0, 0, 4, 0, 0, 0],
    [0, 0, 0, 4, 0, 0, 0, 0, 0],
    [0, 0, 0, 4, 4, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 2, 0, 0, 4],
    [1, 0, 0, 2, 0, 0, 4, 0, 0],
    [1, 1, 0, 2, 2, 0, 4, 4, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0],
], dtype=int)

E3_IN = np.array([
    [2, 0, 0, 4, 0, 0, 8, 0],
    [0, 2, 4, 0, 8, 8, 8, 8],
    [0, 4, 2, 0, 0, 0, 8, 0],
    [4, 0, 0, 2, 0, 0, 8, 0],
], dtype=int)

E3_OUT = np.array([
    [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0],
    [2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4],
    [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0],
    [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0],
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 2, 2, 2, 4, 4, 4, 4, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 4, 4, 4, 4, 2, 2, 2, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 0, 0],
    [0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
    [4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2],
    [0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
    [0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [3, 0, 0, 1],
    [0, 2, 2, 0],
    [0, 2, 2, 0],
    [3, 0, 0, 3],
    [0, 8, 8, 0],
    [8, 8, 8, 8],
    [8, 0, 0, 8],
    [8, 8, 8, 8],
], dtype=int)

T_OUT = np.array([
    [0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
    [3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    [3, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    [3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0],
    [0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0],
    [3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3],
    [3, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 3],
    [3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
 A=min(len(j),len(j[0]));p,c=[r[:A]for r in j[:A]],[r[-A:]for r in j[-A:]]
 if any(max(r)==8 for r in p):p,c=c,p
 return[[p[y//A][x//A]*c[y%A][x%A]//8 for x in range(A*A)]for y in range(A*A)]


# --- Code Golf Solution (Compressed) ---
def q(g):
    R = len(g + g[0]) // 3
    r = range(-R * R, 0)
    return [[sum((g[u % R - t][v % R - t] // 8 * g[u // R + t][v // R + t] for t in (0, R))) for v in r] for u in r]


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

def vconcat(
    a: Grid,
    b: Grid
) -> Grid:
    """ concatenate two grids vertically """
    return a + b

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

def generate_b190f7f5(diff_lb: float, diff_ub: float) -> dict:
    fullcols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 5))
    w = unifint(diff_lb, diff_ub, (2, 5))
    bgc = choice(fullcols)
    cols = remove(bgc, fullcols)
    c = canvas(bgc, (h, w))
    numcd = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numc = choice((numcd, h * w - numcd))
    numc = min(max(1, numc), h * w - 1)
    numcd2 = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numc2 = choice((numcd2, h * w - numcd2))
    numc2 = min(max(2, numc2), h * w - 1)
    inds = totuple(asindices(c))
    srclocs = sample(inds, numc)
    srccol = choice(cols)
    remcols = remove(srccol, cols)
    numcols = unifint(diff_lb, diff_ub, (2, 8))
    trglocs = sample(inds, numc2)
    ccols = sample(remcols, numcols)
    fixc1 = choice(ccols)
    trgobj = [(fixc1, trglocs[0]), (choice(remove(fixc1, ccols)), trglocs[1])] + [(choice(ccols), ij) for ij in trglocs[2:]]
    trgobj = frozenset(trgobj)
    gisrc = fill(c, srccol, srclocs)
    gitrg = paint(c, trgobj)
    catf = choice((hconcat, vconcat))
    ordd = choice(([gisrc, gitrg], [gitrg, gisrc]))
    gi = catf(*ordd)
    go = canvas(bgc, (h**2, w**2))
    for loc in trglocs:
        a, b = loc
        go = fill(go, gitrg[a][b], shift(srclocs, multiply(loc, (h, w))))
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

IntegerSet = FrozenSet[Integer]

Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

ContainerContainer = Container[Container]

ONE = 1

TWO = 2

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

def flip(
    b: Boolean
) -> Boolean:
    """ logical not """
    return not b

def contained(
    value: Any,
    container: Container
) -> Boolean:
    """ element of """
    return value in container

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

def dedupe(
    iterable: Tuple
) -> Tuple:
    """ remove duplicates """
    return tuple(e for i, e in enumerate(iterable) if iterable.index(e) == i)

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

def minimum(
    container: IntegerSet
) -> Integer:
    """ minimum """
    return min(container, default=0)

def argmax(
    container: Container,
    compfunc: Callable
) -> Any:
    """ largest item by custom order """
    return max(container, key=compfunc, default=None)

def argmin(
    container: Container,
    compfunc: Callable
) -> Any:
    """ smallest item by custom order """
    return min(container, key=compfunc, default=None)

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

def ofcolor(
    grid: Grid,
    value: Integer
) -> Indices:
    """ indices of all grid cells with value """
    return frozenset((i, j) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value)

def crop(
    grid: Grid,
    start: IntegerTuple,
    dims: IntegerTuple
) -> Grid:
    """ subgrid specified by start and dimension """
    return tuple(r[start[1]:start[1]+dims[1]] for r in grid[start[0]:start[0]+dims[0]])

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

def asobject(
    grid: Grid
) -> Object:
    """ conversion of grid to object """
    return frozenset((v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r))

def hsplit(
    grid: Grid,
    n: Integer
) -> Tuple:
    """ split grid horizontally """
    h, w = len(grid), len(grid[0]) // n
    offset = len(grid[0]) % n != 0
    return tuple(crop(grid, (0, w * i + i * offset), (h, w)) for i in range(n))

def vsplit(
    grid: Grid,
    n: Integer
) -> Tuple:
    """ split grid vertically """
    h, w = len(grid) // n, len(grid[0])
    offset = len(grid) % n != 0
    return tuple(crop(grid, (h * i + i * offset, 0), (h, w)) for i in range(n))

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_b190f7f5(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = lbind(contained, TWO)
    x1 = lbind(apply, numcolors)
    x2 = compose(x0, x1)
    x3 = lbind(apply, shape)
    x4 = chain(size, dedupe, x3)
    x5 = matcher(x4, ONE)
    x6 = compose(palette, first)
    x7 = compose(palette, last)
    x8 = fork(intersection, x6, x7)
    x9 = compose(size, x8)
    x10 = matcher(x9, ONE)
    x11 = lbind(contained, ONE)
    x12 = compose(minimum, shape)
    x13 = lbind(apply, x12)
    x14 = chain(flip, x11, x13)
    x15 = fork(add, x2, x5)
    x16 = fork(add, x10, x14)
    x17 = fork(add, x15, x16)
    x18 = vsplit(I, TWO)
    x19 = hsplit(I, TWO)
    x20 = astuple(x18, x19)
    x21 = argmax(x20, x17)
    x22 = argmin(x21, numcolors)
    x23 = argmax(x21, numcolors)
    x24 = palette(x22)
    x25 = palette(x23)
    x26 = intersection(x24, x25)
    x27 = first(x26)
    x28 = asindices(x22)
    x29 = ofcolor(x22, x27)
    x30 = difference(x28, x29)
    x31 = asobject(x23)
    x32 = matcher(first, x27)
    x33 = sfilter(x31, x32)
    x34 = difference(x31, x33)
    x35 = shape(x22)
    x36 = multiply(x35, x35)
    x37 = canvas(x27, x36)
    x38 = lbind(shift, x30)
    x39 = lbind(multiply, x35)
    x40 = chain(x38, x39, last)
    x41 = fork(recolor, first, x40)
    x42 = mapply(x41, x34)
    x43 = paint(x37, x42)
    return x43


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_b190f7f5(inp)
        assert pred == _to_grid(expected), f"{name} failed"

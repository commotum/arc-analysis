# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "8e5a5113"
SERIAL = "214"
URL    = "https://arcprize.org/play?task=8e5a5113"

# --- Code Golf Concepts ---
CONCEPTS = [
    "detect_wall",
    "separate_images",
    "image_repetition",
    "image_rotation",
]

# --- Example Grids ---
E1_IN = np.array([
    [1, 1, 2, 5, 0, 0, 0, 5, 0, 0, 0],
    [4, 1, 1, 5, 0, 0, 0, 5, 0, 0, 0],
    [4, 4, 1, 5, 0, 0, 0, 5, 0, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [1, 1, 2, 5, 4, 4, 1, 5, 1, 4, 4],
    [4, 1, 1, 5, 4, 1, 1, 5, 1, 1, 4],
    [4, 4, 1, 5, 1, 1, 2, 5, 2, 1, 1],
], dtype=int)

E2_IN = np.array([
    [6, 3, 3, 5, 0, 0, 0, 5, 0, 0, 0],
    [6, 3, 3, 5, 0, 0, 0, 5, 0, 0, 0],
    [6, 3, 2, 5, 0, 0, 0, 5, 0, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [6, 3, 3, 5, 6, 6, 6, 5, 2, 3, 6],
    [6, 3, 3, 5, 3, 3, 3, 5, 3, 3, 6],
    [6, 3, 2, 5, 2, 3, 3, 5, 3, 3, 6],
], dtype=int)

E3_IN = np.array([
    [2, 7, 8, 5, 0, 0, 0, 5, 0, 0, 0],
    [7, 7, 8, 5, 0, 0, 0, 5, 0, 0, 0],
    [8, 8, 8, 5, 0, 0, 0, 5, 0, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [2, 7, 8, 5, 8, 7, 2, 5, 8, 8, 8],
    [7, 7, 8, 5, 8, 7, 7, 5, 8, 7, 7],
    [8, 8, 8, 5, 8, 8, 8, 5, 8, 7, 2],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [3, 3, 9, 5, 0, 0, 0, 5, 0, 0, 0],
    [9, 9, 9, 5, 0, 0, 0, 5, 0, 0, 0],
    [2, 9, 9, 5, 0, 0, 0, 5, 0, 0, 0],
], dtype=int)

T_OUT = np.array([
    [3, 3, 9, 5, 2, 9, 3, 5, 9, 9, 2],
    [9, 9, 9, 5, 9, 9, 3, 5, 9, 9, 9],
    [2, 9, 9, 5, 9, 9, 9, 5, 9, 3, 3],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(g,R=range):
 A=[[c for c in r[:3]] for r in g]
 C=[r[::-1]for r in A[::-1]]
 for r in R(3):
  for c in R(3):
   g[r][c+4]=A[-(c+1)][r];g[r][c+8]=C[r][c]
 return g


# --- Code Golf Solution (Compressed) ---
def q(g):
    return [r + g.pop()[3::-1] for r, *r[4:] in zip(g * 1, *g[::-1])]


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

def hconcat(
    a: Grid,
    b: Grid
) -> Grid:
    """ concatenate two grids horizontally """
    return tuple(i + j for i, j in zip(a, b))

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

def generate_8e5a5113(diff_lb: float, diff_ub: float) -> dict:
    colopts = interval(0, 10, 1)
    d = unifint(diff_lb, diff_ub, (2, 9))
    bgc = choice(colopts)
    remcols = remove(bgc, colopts)
    k = 4 if d < 7 else 3
    nbound = (2, k)
    num = unifint(diff_lb, diff_ub, nbound)
    rotfs = (identity, rot90, rot180, rot270)
    barc = choice(remcols)
    remcols = remove(barc, remcols)
    colbnds = (1, 8)
    ncols = unifint(diff_lb, diff_ub, colbnds)
    patcols = sample(remcols, ncols)
    bgcanv = canvas(bgc, (d, d))
    c = canvas(bgc, (d, d))
    inds = totuple(asindices(c))
    ncolbnds = (1, d ** 2 - 1)
    ncells = unifint(diff_lb, diff_ub, ncolbnds)
    indsss = sample(inds, ncells)
    for ij in indsss:
        c = fill(c, choice(patcols), {ij})
    barr = canvas(barc, (d, 1))
    fillinidx = choice(interval(0, num, 1))
    gi = rot90(rot270(c if fillinidx == 0 else bgcanv))
    go = rot90(rot270(c))
    for j in range(num - 1):
        c = rot90(c)
        gi = hconcat(hconcat(gi, barr), c if j + 1 == fillinidx else bgcanv)
        go = hconcat(hconcat(go, barr), c)
    if choice((True, False)):
        gi = rot90(gi)
        go = rot90(go)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Numerical = Union[Integer, IntegerTuple]

IntegerSet = FrozenSet[Integer]

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

ContainerContainer = Container[Container]

ONE = 1

FOUR = 4

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

def combine(
    a: Container,
    b: Container
) -> Container:
    """ union """
    return type(a)((*a, *b))

def merge(
    containers: ContainerContainer
) -> Container:
    """ merging """
    return type(containers)(e for c in containers for e in c)

def argmax(
    container: Container,
    compfunc: Callable
) -> Any:
    """ largest item by custom order """
    return max(container, key=compfunc, default=None)

def initset(
    value: Any
) -> FrozenSet:
    """ initialize container """
    return frozenset({value})

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

def first(
    container: Container
) -> Any:
    """ first item of container """
    return next(iter(container))

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

def rapply(
    functions: Container,
    value: Any
) -> Container:
    """ apply each function in container to value """
    return type(functions)(function(value) for function in functions)

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

def portrait(
    piece: Piece
) -> Boolean:
    """ whether height is greater than width """
    return height(piece) > width(piece)

def ulcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of upper left corner """
    return tuple(map(min, zip(*toindices(patch))))

def crop(
    grid: Grid,
    start: IntegerTuple,
    dims: IntegerTuple
) -> Grid:
    """ subgrid specified by start and dimension """
    return tuple(r[start[1]:start[1]+dims[1]] for r in grid[start[0]:start[0]+dims[0]])

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

def subgrid(
    patch: Patch,
    grid: Grid
) -> Grid:
    """ smallest subgrid containing object """
    return crop(grid, ulcorner(patch), shape(patch))

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

def verify_8e5a5113(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = portrait(I)
    x1 = branch(x0, identity, rot90)
    x2 = branch(x0, identity, rot270)
    x3 = x1(I)
    x4 = width(x3)
    x5 = toivec(x4)
    x6 = index(x3, x5)
    x7 = shape(x3)
    x8 = canvas(x6, x7)
    x9 = hconcat(x3, x8)
    x10 = objects(x9, F, T, T)
    x11 = argmax(x10, numcolors)
    x12 = subgrid(x11, x3)
    x13 = interval(ONE, FOUR, ONE)
    x14 = lbind(power, rot90)
    x15 = lbind(power, rot270)
    x16 = rbind(rapply, x12)
    x17 = compose(initset, x14)
    x18 = chain(first, x16, x17)
    x19 = rbind(rapply, x12)
    x20 = compose(initset, x15)
    x21 = chain(first, x19, x20)
    x22 = compose(asobject, x18)
    x23 = uppermost(x11)
    x24 = lbind(add, x23)
    x25 = height(x11)
    x26 = increment(x25)
    x27 = lbind(multiply, x26)
    x28 = chain(toivec, x24, x27)
    x29 = fork(shift, x22, x28)
    x30 = compose(asobject, x21)
    x31 = uppermost(x11)
    x32 = lbind(subtract, x31)
    x33 = height(x11)
    x34 = increment(x33)
    x35 = lbind(multiply, x34)
    x36 = chain(toivec, x32, x35)
    x37 = fork(shift, x30, x36)
    x38 = fork(combine, x29, x37)
    x39 = mapply(x38, x13)
    x40 = paint(x3, x39)
    x41 = x2(x40)
    return x41


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_8e5a5113(inp)
        assert pred == _to_grid(expected), f"{name} failed"

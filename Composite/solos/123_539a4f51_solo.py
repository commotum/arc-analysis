# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "539a4f51"
SERIAL = "123"
URL    = "https://arcprize.org/play?task=539a4f51"

# --- Code Golf Concepts ---
CONCEPTS = [
    "pattern_expansion",
    "image_expansion",
]

# --- Example Grids ---
E1_IN = np.array([
    [2, 2, 2, 3, 0],
    [2, 2, 2, 3, 0],
    [2, 2, 2, 3, 0],
    [3, 3, 3, 3, 0],
    [0, 0, 0, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [2, 2, 2, 3, 2, 2, 2, 3, 2, 2],
    [2, 2, 2, 3, 2, 2, 2, 3, 2, 2],
    [2, 2, 2, 3, 2, 2, 2, 3, 2, 2],
    [3, 3, 3, 3, 2, 2, 2, 3, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 3, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 3, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 3, 2, 2],
    [3, 3, 3, 3, 3, 3, 3, 3, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
], dtype=int)

E2_IN = np.array([
    [1, 1, 4, 6, 0],
    [1, 1, 4, 6, 0],
    [4, 4, 4, 6, 0],
    [6, 6, 6, 6, 0],
    [0, 0, 0, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [1, 1, 4, 6, 1, 1, 4, 6, 1, 1],
    [1, 1, 4, 6, 1, 1, 4, 6, 1, 1],
    [4, 4, 4, 6, 1, 1, 4, 6, 1, 1],
    [6, 6, 6, 6, 1, 1, 4, 6, 1, 1],
    [1, 1, 1, 1, 1, 1, 4, 6, 1, 1],
    [1, 1, 1, 1, 1, 1, 4, 6, 1, 1],
    [4, 4, 4, 4, 4, 4, 4, 6, 1, 1],
    [6, 6, 6, 6, 6, 6, 6, 6, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
], dtype=int)

E3_IN = np.array([
    [2, 3, 4, 1, 6],
    [3, 3, 4, 1, 6],
    [4, 4, 4, 1, 6],
    [1, 1, 1, 1, 6],
    [6, 6, 6, 6, 6],
], dtype=int)

E3_OUT = np.array([
    [2, 3, 4, 1, 6, 2, 3, 4, 1, 6],
    [3, 3, 4, 1, 6, 2, 3, 4, 1, 6],
    [4, 4, 4, 1, 6, 2, 3, 4, 1, 6],
    [1, 1, 1, 1, 6, 2, 3, 4, 1, 6],
    [6, 6, 6, 6, 6, 2, 3, 4, 1, 6],
    [2, 2, 2, 2, 2, 2, 3, 4, 1, 6],
    [3, 3, 3, 3, 3, 3, 3, 4, 1, 6],
    [4, 4, 4, 4, 4, 4, 4, 4, 1, 6],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 6],
    [6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [7, 7, 3, 2, 2],
    [7, 7, 3, 2, 2],
    [3, 3, 3, 2, 2],
    [2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2],
], dtype=int)

T_OUT = np.array([
    [7, 7, 3, 2, 2, 7, 7, 3, 2, 2],
    [7, 7, 3, 2, 2, 7, 7, 3, 2, 2],
    [3, 3, 3, 2, 2, 7, 7, 3, 2, 2],
    [2, 2, 2, 2, 2, 7, 7, 3, 2, 2],
    [2, 2, 2, 2, 2, 7, 7, 3, 2, 2],
    [7, 7, 7, 7, 7, 7, 7, 3, 2, 2],
    [7, 7, 7, 7, 7, 7, 7, 3, 2, 2],
    [3, 3, 3, 3, 3, 3, 3, 3, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(g,R=range):
 g=[[x for x in r if x>0] for r in g if r.count(0)<2]
 g=[[r[0]]*10 for r in g+g+g]
 for r in R(10):
  for c in R(10):g[r][c]=g[c][r]
 return g[:10]


# --- Code Golf Solution (Compressed) ---
def q(g):
    return [[(a := (g[0][:4 + any(g[4])] * 3))[i]] * i + a[i:10] for i in range(10)]


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

def generate_539a4f51(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(1, 10, 1)
    d = unifint(diff_lb, diff_ub, (2, 15))
    h, w = d, d
    gi = canvas(0, (h, w))
    numc = unifint(diff_lb, diff_ub, (2, 9))
    ccols = sample(cols, numc)
    numocc = unifint(diff_lb, diff_ub, (1, d))
    arr = [choice(ccols) for k in range(numocc)]
    while len(set(arr)) == 1:
        arr = [choice(ccols) for k in range(d)]
    for j, col in enumerate(arr):
        gi = fill(gi, col, connect((j, 0), (j, j)) | connect((0, j), (j, j)))
    go = canvas(0, (2*d, 2*d))
    for j in range(2*d):
        col = arr[j % len(arr)]
        go = fill(go, col, connect((j, 0), (j, j)) | connect((0, j), (j, j)))
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Numerical = Union[Integer, IntegerTuple]

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

ContainerContainer = Container[Container]

ZERO = 0

TWO = 2

F = False

T = True

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

def repeat(
    item: Any,
    num: Integer
) -> Tuple:
    """ repetition of item within vector """
    return tuple(item for i in range(num))

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

def rapply(
    functions: Container,
    value: Any
) -> Container:
    """ apply each function in container to value """
    return type(functions)(function(value) for function in functions)

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

def asobject(
    grid: Grid
) -> Object:
    """ conversion of grid to object """
    return frozenset((v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r))

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

def upscale(
    element: Element,
    factor: Integer
) -> Element:
    """ upscale object or grid """
    if isinstance(element, tuple):
        upscaled_grid = tuple()
        for row in element:
            upscaled_row = tuple()
            for value in row:
                upscaled_row = upscaled_row + tuple(value for num in range(factor))
            upscaled_grid = upscaled_grid + tuple(upscaled_row for num in range(factor))
        return upscaled_grid
    else:
        if len(element) == 0:
            return frozenset()
        di_inv, dj_inv = ulcorner(element)
        di, dj = (-di_inv, -dj_inv)
        normed_obj = shift(element, (di, dj))
        upscaled_obj = set()
        for value, (i, j) in normed_obj:
            for io in range(factor):
                for jo in range(factor):
                    upscaled_obj.add((value, (i * factor + io, j * factor + jo)))
        return shift(frozenset(upscaled_obj), (di_inv, dj_inv))

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_539a4f51(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = astuple(identity, cmirror)
    x1 = astuple(hmirror, vmirror)
    x2 = combine(x0, x1)
    x3 = fork(multiply, height, width)
    x4 = rbind(objects, F)
    x5 = rbind(x4, F)
    x6 = rbind(x5, T)
    x7 = rbind(argmin, x3)
    x8 = lbind(contained, ORIGIN)
    x9 = chain(x8, toindices, x7)
    x10 = compose(x9, x6)
    x11 = lbind(compose, x10)
    x12 = rbind(rapply, I)
    x13 = compose(initset, x11)
    x14 = chain(first, x12, x13)
    x15 = extract(x2, x14)
    x16 = x15(I)
    x17 = height(I)
    x18 = first(x16)
    x19 = matcher(identity, ZERO)
    x20 = compose(flip, x19)
    x21 = sfilter(x18, x20)
    x22 = size(x21)
    x23 = divide(x17, x22)
    x24 = increment(x23)
    x25 = double(x24)
    x26 = repeat(x21, x25)
    x27 = merge(x26)
    x28 = double(x17)
    x29 = repeat(x27, x28)
    x30 = asobject(x29)
    x31 = chain(increment, last, last)
    x32 = compose(first, last)
    x33 = fork(greater, x31, x32)
    x34 = sfilter(x30, x33)
    x35 = upscale(x16, TWO)
    x36 = dmirror(x34)
    x37 = combine(x34, x36)
    x38 = paint(x35, x37)
    x39 = x15(x38)
    return x39


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_539a4f51(inp)
        assert pred == _to_grid(expected), f"{name} failed"

# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "5614dbcf"
SERIAL = "130"
URL    = "https://arcprize.org/play?task=5614dbcf"

# --- Code Golf Concepts ---
CONCEPTS = [
    "remove_noise",
    "image_resizing",
]

# --- Example Grids ---
E1_IN = np.array([
    [3, 3, 3, 0, 0, 0, 8, 8, 8],
    [3, 3, 3, 0, 0, 0, 8, 5, 8],
    [3, 3, 3, 0, 0, 0, 8, 8, 8],
    [0, 0, 0, 7, 5, 7, 0, 0, 0],
    [0, 0, 0, 7, 7, 7, 0, 0, 0],
    [0, 0, 0, 7, 7, 7, 0, 0, 0],
    [6, 6, 6, 0, 0, 5, 9, 9, 9],
    [6, 6, 6, 0, 0, 0, 9, 9, 9],
    [6, 5, 6, 0, 5, 0, 9, 9, 5],
], dtype=int)

E1_OUT = np.array([
    [3, 0, 8],
    [0, 7, 0],
    [6, 0, 9],
], dtype=int)

E2_IN = np.array([
    [0, 0, 0, 2, 2, 2, 0, 0, 0],
    [0, 5, 0, 2, 2, 2, 0, 0, 0],
    [0, 0, 0, 2, 2, 2, 0, 0, 0],
    [5, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 5, 0, 7, 7, 7, 0, 0, 0],
    [0, 0, 0, 7, 7, 5, 0, 0, 0],
    [0, 0, 0, 7, 7, 7, 0, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [0, 2, 0],
    [0, 0, 0],
    [0, 7, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [4, 4, 4, 0, 0, 0, 0, 5, 0],
    [5, 4, 4, 0, 0, 0, 0, 0, 0],
    [4, 4, 4, 0, 5, 0, 0, 0, 0],
    [0, 0, 0, 3, 3, 3, 0, 5, 0],
    [0, 0, 0, 3, 3, 3, 0, 0, 0],
    [0, 0, 0, 3, 3, 3, 0, 0, 0],
    [0, 0, 5, 9, 9, 9, 0, 0, 0],
    [0, 0, 0, 9, 5, 9, 0, 0, 0],
    [0, 0, 0, 9, 9, 9, 0, 0, 0],
], dtype=int)

T_OUT = np.array([
    [4, 0, 0],
    [0, 3, 0],
    [0, 9, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
 A=range
 c=[[0]*3for _ in A(3)]
 for E in A(3):
  for k in A(3):
   W={}
   for l in A(3):
    for J in A(3):a=j[E*3+l][k*3+J];W[a]=W.get(a,0)+1
   c[E][k]=max(W,key=W.get)
 return c


# --- Code Golf Solution (Compressed) ---
def q(g):
    return g * (g != 5) and (g * -1 * -1 or [max(map(p, g[:3]))] + p(g[3:]))


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

Element = Union[Object, Grid]

THREE = 3

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

def initset(
    value: Any
) -> FrozenSet:
    """ initialize container """
    return frozenset({value})

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

def generate_5614dbcf(diff_lb: float, diff_ub: float) -> dict:
    dim_bounds = (2, 10)
    col_card_bounds = (1, 8)
    noise_card_bounds = (0, 8)
    colopts = remove(5, interval(1, 10, 1))
    noisedindscands = totuple(asindices(canvas(0, (3, 3))))
    d = unifint(diff_lb, diff_ub, dim_bounds)
    cells_card_bounds = (1, d * d)
    go = canvas(0, (d, d))
    inds = totuple(asindices(go))
    numocc = unifint(diff_lb, diff_ub, cells_card_bounds)
    numcol = unifint(diff_lb, diff_ub, col_card_bounds)
    occs = sample(inds, numocc)
    colset = sample(colopts, numcol)
    gi = upscale(go, THREE)
    for occ in inds:
        offset = multiply(3, occ)
        numnoise = unifint(diff_lb, diff_ub, noise_card_bounds)
        noise = sample(noisedindscands, numnoise)
        if occ in occs:
            col = choice(colset)
            go = fill(go, col, initset(occ))
            gi = fill(gi, col, shift(noisedindscands, offset))
        gi = fill(gi, 5, shift(noise, offset))
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Piece = Union[Grid, Patch]

ContainerContainer = Container[Container]

ZERO = 0

ONE = 1

FIVE = 5

THREE_BY_THREE = (3, 3)

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

def flip(
    b: Boolean
) -> Boolean:
    """ logical not """
    return not b

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

def last(
    container: Container
) -> Any:
    """ last item of container """
    return max(enumerate(container))[1]

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

def downscale(
    grid: Grid,
    factor: Integer
) -> Grid:
    """ downscale grid """
    h, w = len(grid), len(grid[0])
    downscaled_grid = tuple()
    for i in range(h):
        downscaled_row = tuple()
        for j in range(w):
            if j % factor == 0:
                downscaled_row = downscaled_row + (grid[i][j],)
        downscaled_grid = downscaled_grid + (downscaled_row, )
    h = len(downscaled_grid)
    downscaled_grid2 = tuple()
    for i in range(h):
        if i % factor == 0:
            downscaled_grid2 = downscaled_grid2 + (downscaled_grid[i],)
    return downscaled_grid2

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_5614dbcf(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = canvas(ZERO, THREE_BY_THREE)
    x1 = asindices(x0)
    x2 = shape(I)
    x3 = divide(x2, THREE)
    x4 = first(x3)
    x5 = last(x3)
    x6 = interval(ZERO, x4, ONE)
    x7 = interval(ZERO, x5, ONE)
    x8 = product(x6, x7)
    x9 = rbind(multiply, THREE)
    x10 = apply(x9, x8)
    x11 = matcher(first, FIVE)
    x12 = compose(flip, x11)
    x13 = rbind(sfilter, x12)
    x14 = rbind(toobject, I)
    x15 = lbind(shift, x1)
    x16 = chain(x13, x14, x15)
    x17 = compose(color, x16)
    x18 = lbind(shift, x1)
    x19 = fork(recolor, x17, x18)
    x20 = mapply(x19, x10)
    x21 = paint(I, x20)
    x22 = downscale(x21, THREE)
    return x22


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_5614dbcf(inp)
        assert pred == _to_grid(expected), f"{name} failed"

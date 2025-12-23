# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "2204b7a8"
SERIAL = "040"
URL    = "https://arcprize.org/play?task=2204b7a8"

# --- Code Golf Concepts ---
CONCEPTS = [
    "proximity_guessing",
    "recoloring",
]

# --- Example Grids ---
E1_IN = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [1, 0, 0, 0, 0, 0, 3, 0, 0, 2],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [1, 0, 0, 0, 3, 0, 0, 0, 0, 2],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [1, 0, 3, 0, 0, 0, 0, 0, 0, 2],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 2],
], dtype=int)

E1_OUT = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [1, 0, 0, 0, 0, 0, 2, 0, 0, 2],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 2],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 2],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 2],
], dtype=int)

E2_IN = np.array([
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 3, 0, 0, 0, 0, 0, 3, 0, 0],
    [0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 3, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
], dtype=int)

E2_OUT = np.array([
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 4, 0, 0, 0, 0, 0, 4, 0, 0],
    [0, 0, 0, 4, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 7, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 7, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
], dtype=int)

E3_IN = np.array([
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
    [0, 3, 0, 0, 0, 0, 0, 3, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 3, 0, 0, 0, 0, 3, 0, 0, 0],
    [0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
], dtype=int)

E3_OUT = np.array([
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 0],
    [0, 8, 0, 0, 0, 0, 0, 8, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 9, 0, 0, 0, 0, 9, 0, 0, 0],
    [0, 0, 0, 9, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [5, 3, 0, 0, 0, 0, 0, 0, 0, 4],
    [5, 0, 0, 0, 0, 3, 0, 0, 3, 4],
    [5, 0, 0, 0, 0, 0, 0, 0, 0, 4],
    [5, 0, 0, 3, 0, 0, 0, 0, 0, 4],
    [5, 0, 0, 0, 0, 0, 3, 0, 0, 4],
    [5, 0, 0, 3, 0, 0, 0, 0, 0, 4],
    [5, 0, 0, 0, 0, 0, 0, 0, 0, 4],
    [5, 0, 0, 0, 3, 0, 0, 0, 0, 4],
    [5, 0, 3, 0, 0, 0, 3, 0, 0, 4],
    [5, 0, 0, 0, 0, 0, 0, 0, 0, 4],
], dtype=int)

T_OUT = np.array([
    [5, 5, 0, 0, 0, 0, 0, 0, 0, 4],
    [5, 0, 0, 0, 0, 4, 0, 0, 4, 4],
    [5, 0, 0, 0, 0, 0, 0, 0, 0, 4],
    [5, 0, 0, 5, 0, 0, 0, 0, 0, 4],
    [5, 0, 0, 0, 0, 0, 4, 0, 0, 4],
    [5, 0, 0, 5, 0, 0, 0, 0, 0, 4],
    [5, 0, 0, 0, 0, 0, 0, 0, 0, 4],
    [5, 0, 0, 0, 5, 0, 0, 0, 0, 4],
    [5, 0, 5, 0, 0, 0, 4, 0, 0, 4],
    [5, 0, 0, 0, 0, 0, 0, 0, 0, 4],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
	A=range;c=[J[:]for J in j];E=j[0][0]==j[0][9];k,W=(j[0][0],j[9][0])if E else(j[0][0],j[0][9]);l=next(J for a in j for J in a if J and J not in[k,W])
	for J in A(10):
		for a in A(10):
			if j[J][a]==l:C=(J,9-J)if E else(a,9-a);c[J][a]=k if C[0]<C[1]else W
	return c


# --- Code Golf Solution (Compressed) ---
def q(g, h=[]):
    return g * 0 != 0 and [*map(p, g[:1] * 5 + g[9:] * 5, h + g)] or h % ~h & g


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, sample, uniform

Integer = int

IntegerTuple = Tuple[Integer, Integer]

IntegerSet = FrozenSet[Integer]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

ORIGIN = (0, 0)

RIGHT = (0, 1)

def difference(
    a: Container,
    b: Container
) -> Container:
    """ difference """
    return type(a)(e for e in a if e not in b)

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

def totuple(
    container: FrozenSet
) -> Tuple:
    """ conversion to tuple """
    return tuple(container)

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

def palette(
    element: Element
) -> IntegerSet:
    """ colors occurring in object or grid """
    if isinstance(element, tuple):
        return frozenset({v for r in element for v in r})
    return frozenset({v for v, _ in element})

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

def vfrontier(
    location: IntegerTuple
) -> Indices:
    """ vertical frontier """
    return frozenset((i, location[1]) for i in range(30))

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

def generate_2204b7a8(diff_lb: float, diff_ub: float) -> dict:
    dim_bounds = (4, 30)
    colopts = interval(0, 10, 1)
    while True:
        h = unifint(diff_lb, diff_ub, dim_bounds)
        w = unifint(diff_lb, diff_ub, dim_bounds)
        bgc = choice(colopts)
        remcols = remove(bgc, colopts)
        c = canvas(bgc, (h, w))
        inds = totuple(shift(asindices(canvas(0, (h, w - 2))), RIGHT))
        ccol = choice(remcols)
        remcols2 = remove(ccol, remcols)
        c1 = choice(remcols2)
        c2 = choice(remove(c1, remcols2))
        nc_bounds = (1, (h * (w - 2)) // 2 - 1)
        nc = unifint(diff_lb, diff_ub, nc_bounds)
        locs = sample(inds, nc)
        if w % 2 == 1:
            locs = difference(locs, vfrontier(tojvec(w // 2)))
        gi = fill(c, c1, vfrontier(ORIGIN))
        gi = fill(gi, c2, vfrontier(tojvec(w - 1)))
        gi = fill(gi, ccol, locs)
        a = sfilter(locs, lambda ij: last(ij) < w // 2)
        b = difference(locs, a)
        go = fill(gi, c1, a)
        go = fill(go, c2, b)
        if len(palette(gi)) == 4:
            break
    if choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Numerical = Union[Integer, IntegerTuple]

ONE = 1

def identity(
    x: Any
) -> Any:
    """ identity function """
    return x

def even(
    n: Integer
) -> Boolean:
    """ evenness """
    return n % 2 == 0

def flip(
    b: Boolean
) -> Boolean:
    """ logical not """
    return not b

def equality(
    a: Any,
    b: Any
) -> Boolean:
    """ equality """
    return a == b

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

def decrement(
    x: Numerical
) -> Numerical:
    """ decrementing """
    return x - 1 if isinstance(x, int) else (x[0] - 1, x[1] - 1)

def first(
    container: Container
) -> Any:
    """ first item of container """
    return next(iter(container))

def astuple(
    a: Integer,
    b: Integer
) -> IntegerTuple:
    """ constructs a tuple """
    return (a, b)

def branch(
    condition: Boolean,
    if_value: Any,
    else_value: Any
) -> Any:
    """ if else branching """
    return if_value if condition else else_value

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

def rot90(
    grid: Grid
) -> Grid:
    """ quarter clockwise rotation """
    return tuple(row for row in zip(*grid[::-1]))

def rot270(
    grid: Grid
) -> Grid:
    """ quarter anticlockwise rotation """
    return tuple(tuple(row[::-1]) for row in zip(*grid[::-1]))[::-1]

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

def replace(
    grid: Grid,
    replacee: Integer,
    replacer: Integer
) -> Grid:
    """ color substitution """
    return tuple(tuple(replacer if v == replacee else v for v in r) for r in grid)

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

def tophalf(
    grid: Grid
) -> Grid:
    """ upper half of grid """
    return grid[:len(grid) // 2]

def bottomhalf(
    grid: Grid
) -> Grid:
    """ lower half of grid """
    return grid[len(grid) // 2 + len(grid) % 2:]

def lefthalf(
    grid: Grid
) -> Grid:
    """ left half of grid """
    return rot270(tophalf(rot90(grid)))

def righthalf(
    grid: Grid
) -> Grid:
    """ right half of grid """
    return rot270(bottomhalf(rot90(grid)))

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_2204b7a8(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = first(I)
    x1 = dedupe(x0)
    x2 = size(x1)
    x3 = equality(x2, ONE)
    x4 = flip(x3)
    x5 = branch(x4, lefthalf, tophalf)
    x6 = branch(x4, righthalf, bottomhalf)
    x7 = branch(x4, hconcat, vconcat)
    x8 = x5(I)
    x9 = x6(I)
    x10 = index(x8, ORIGIN)
    x11 = shape(x9)
    x12 = decrement(x11)
    x13 = index(x9, x12)
    x14 = mostcolor(I)
    x15 = mostcolor(I)
    x16 = palette(I)
    x17 = remove(x10, x16)
    x18 = remove(x13, x17)
    x19 = remove(x15, x18)
    x20 = first(x19)
    x21 = replace(x8, x20, x10)
    x22 = branch(x4, dmirror, identity)
    x23 = branch(x4, height, width)
    x24 = x23(I)
    x25 = astuple(ONE, x24)
    x26 = canvas(x14, x25)
    x27 = x22(x26)
    x28 = replace(x9, x20, x13)
    x29 = x7(x21, x27)
    x30 = branch(x4, width, height)
    x31 = x30(I)
    x32 = even(x31)
    x33 = branch(x32, x21, x29)
    x34 = x7(x33, x28)
    return x34


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_2204b7a8(inp)
        assert pred == _to_grid(expected), f"{name} failed"

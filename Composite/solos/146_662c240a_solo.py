# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "662c240a"
SERIAL = "146"
URL    = "https://arcprize.org/play?task=662c240a"

# --- Code Golf Concepts ---
CONCEPTS = [
    "separate_images",
    "detect_symmetry",
    "find_the_intruder",
    "crop",
]

# --- Example Grids ---
E1_IN = np.array([
    [8, 9, 8],
    [9, 8, 8],
    [8, 8, 8],
    [2, 2, 1],
    [2, 2, 1],
    [1, 1, 2],
    [4, 4, 4],
    [4, 4, 3],
    [3, 3, 3],
], dtype=int)

E1_OUT = np.array([
    [4, 4, 4],
    [4, 4, 3],
    [3, 3, 3],
], dtype=int)

E2_IN = np.array([
    [1, 5, 5],
    [5, 1, 1],
    [5, 1, 1],
    [3, 3, 3],
    [3, 6, 3],
    [3, 6, 6],
    [7, 7, 7],
    [7, 2, 2],
    [7, 2, 2],
], dtype=int)

E2_OUT = np.array([
    [3, 3, 3],
    [3, 6, 3],
    [3, 6, 6],
], dtype=int)

E3_IN = np.array([
    [2, 2, 2],
    [2, 2, 3],
    [2, 3, 3],
    [5, 7, 7],
    [7, 5, 5],
    [7, 5, 5],
    [8, 8, 1],
    [1, 8, 1],
    [1, 8, 1],
], dtype=int)

E3_OUT = np.array([
    [8, 8, 1],
    [1, 8, 1],
    [1, 8, 1],
], dtype=int)

E4_IN = np.array([
    [8, 8, 4],
    [4, 4, 4],
    [4, 4, 8],
    [1, 1, 3],
    [1, 3, 3],
    [3, 3, 1],
    [6, 2, 2],
    [2, 2, 2],
    [2, 2, 6],
], dtype=int)

E4_OUT = np.array([
    [8, 8, 4],
    [4, 4, 4],
    [4, 4, 8],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [5, 4, 4],
    [4, 5, 4],
    [4, 5, 4],
    [3, 3, 2],
    [3, 3, 2],
    [2, 2, 3],
    [1, 1, 1],
    [1, 8, 8],
    [1, 8, 8],
], dtype=int)

T_OUT = np.array([
    [5, 4, 4],
    [4, 5, 4],
    [4, 5, 4],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(g, R=range):
    return [[[g[k + i][j] for j in R(3)] for i in R(3)] for k in R(0, 9, 3) if [[g[k + i][j] for j in R(3)] for i in R(3)] != [[g[k + j][i] for j in R(3)] for i in R(3)]][0]


# --- Code Golf Solution (Compressed) ---
def q(m):
    return (a := m[:3]) * (a != [*map(list, zip(*a))]) or p(m[3:])


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

Piece = Union[Grid, Patch]

def difference(
    a: Container,
    b: Container
) -> Container:
    """ difference """
    return type(a)(e for e in a if e not in b)

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

def toindices(
    patch: Patch
) -> Indices:
    """ indices of object cells """
    if len(patch) == 0:
        return frozenset()
    if isinstance(next(iter(patch))[1], tuple):
        return frozenset(index for value, index in patch)
    return patch

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

def generate_662c240a(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    d = unifint(diff_lb, diff_ub, (2, 7))
    ng = unifint(diff_lb, diff_ub, (2, 30 // d))
    nc = unifint(diff_lb, diff_ub, (2, min(9, d ** 2)))
    c = canvas(-1, (d, d))
    inds = totuple(asindices(c))
    tria = sfilter(inds, lambda ij: ij[1] >= ij[0])
    tcolset = sample(cols, nc)
    triaf = frozenset((choice(tcolset), ij) for ij in tria)
    triaf = triaf | dmirror(triaf)
    gik = paint(c, triaf)
    ndistinv = unifint(diff_lb, diff_ub, (0, (d * (d - 1) // 2 - 1)))
    ndist = d * (d - 1) // 2 - ndistinv
    distinds = sample(difference(inds, sfilter(inds, lambda ij: ij[0] == ij[1])), ndist)
    
    for ij in distinds:
        if gik[ij[0]][ij[1]] == gik[ij[1]][ij[0]]:
            gik = fill(gik, choice(remove(gik[ij[0]][ij[1]], tcolset)), {ij})
        else:
            gik = fill(gik, gik[ij[1]][ij[0]], {ij})
    gi = gik
    go = tuple(e for e in gik)
    concatf = choice((hconcat, vconcat))
    for k in range(ng - 1):
        tria = sfilter(inds, lambda ij: ij[1] >= ij[0])
        tcolset = sample(cols, nc)
        triaf = frozenset((choice(tcolset), ij) for ij in tria)
        triaf = triaf | dmirror(triaf)
        gik = paint(c, triaf)
        if choice((True, False)):
            gi = concatf(gi, gik)
        else:
            gi = concatf(gik, gi)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Numerical = Union[Integer, IntegerTuple]

IntegerSet = FrozenSet[Integer]

def identity(
    x: Any
) -> Any:
    """ identity function """
    return x

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

def equality(
    a: Any,
    b: Any
) -> Boolean:
    """ equality """
    return a == b

def maximum(
    container: IntegerSet
) -> Integer:
    """ maximum """
    return max(container, default=0)

def minimum(
    container: IntegerSet
) -> Integer:
    """ minimum """
    return min(container, default=0)

def extract(
    container: Container,
    condition: Callable
) -> Any:
    """ first element of container that satisfies condition """
    return next(e for e in container if condition(e))

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

def portrait(
    piece: Piece
) -> Boolean:
    """ whether height is greater than width """
    return height(piece) > width(piece)

def crop(
    grid: Grid,
    start: IntegerTuple,
    dims: IntegerTuple
) -> Grid:
    """ subgrid specified by start and dimension """
    return tuple(r[start[1]:start[1]+dims[1]] for r in grid[start[0]:start[0]+dims[0]])

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

def verify_662c240a(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = portrait(I)
    x1 = branch(x0, vsplit, hsplit)
    x2 = shape(I)
    x3 = maximum(x2)
    x4 = minimum(x2)
    x5 = divide(x3, x4)
    x6 = x1(I, x5)
    x7 = fork(equality, identity, dmirror)
    x8 = compose(flip, x7)
    x9 = extract(x6, x8)
    return x9


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("E4", E4_IN, E4_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_662c240a(inp)
        assert pred == _to_grid(expected), f"{name} failed"

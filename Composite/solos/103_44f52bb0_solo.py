# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "44f52bb0"
SERIAL = "103"
URL    = "https://arcprize.org/play?task=44f52bb0"

# --- Code Golf Concepts ---
CONCEPTS = [
    "detect_symmetry",
    "associate_images_to_bools",
]

# --- Example Grids ---
E1_IN = np.array([
    [2, 0, 2],
    [0, 2, 0],
    [2, 0, 2],
], dtype=int)

E1_OUT = np.array([
    [1],
], dtype=int)

E2_IN = np.array([
    [2, 0, 0],
    [2, 0, 0],
    [0, 2, 0],
], dtype=int)

E2_OUT = np.array([
    [7],
], dtype=int)

E3_IN = np.array([
    [2, 0, 2],
    [2, 0, 2],
    [2, 0, 2],
], dtype=int)

E3_OUT = np.array([
    [1],
], dtype=int)

E4_IN = np.array([
    [0, 0, 0],
    [2, 0, 2],
    [0, 0, 0],
], dtype=int)

E4_OUT = np.array([
    [1],
], dtype=int)

E5_IN = np.array([
    [2, 2, 0],
    [0, 2, 2],
    [0, 0, 0],
], dtype=int)

E5_OUT = np.array([
    [7],
], dtype=int)

E6_IN = np.array([
    [2, 2, 0],
    [0, 2, 0],
    [0, 0, 0],
], dtype=int)

E6_OUT = np.array([
    [7],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [2, 0, 2],
    [2, 2, 2],
    [2, 0, 2],
], dtype=int)

T_OUT = np.array([
    [1],
], dtype=int)

T2_IN = np.array([
    [0, 0, 0],
    [2, 0, 0],
    [2, 0, 0],
], dtype=int)

T2_OUT = np.array([
    [7],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
    return [[1 if [j[i][0] for i in range(3)] == [j[i][2] for i in range(3)] else 7]]


# --- Code Golf Solution (Compressed) ---
def q(m):
    return [[m[0] == m[2] or 7]]


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

def insert(
    value: Any,
    container: FrozenSet
) -> FrozenSet:
    """ insert item into container """
    return container.union(frozenset({value}))

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

def generate_44f52bb0(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    ncols = unifint(diff_lb, diff_ub, (2, 9))
    ccols = sample(remcols, ncols)
    gi = canvas(bgc, (h, w))
    numcells = unifint(diff_lb, diff_ub, (1, h * w - 1))
    inds = asindices(gi)
    while gi == hmirror(gi):
        cells = sample(totuple(inds), numcells)
        gi = canvas(bgc, (h, w))
        for ij in cells:
            a, b = ij
            col = choice(ccols)
            gi = fill(gi, col, {ij})
            gi = fill(gi, col, {(a, w - 1 - b)})
    issymm = choice((True, False))
    if not issymm:
        numpert = unifint(diff_lb, diff_ub, (1, h * (w // 2)))
        cands = asindices(canvas(-1, (h, w // 2)))
        locs = sample(totuple(cands), numpert)
        for a, b in locs:
            col = gi[a][b]
            newcol = choice(totuple(remove(col, insert(bgc, set(ccols)))))
            gi = fill(gi, newcol, {(a, b)})
        go = canvas(7, (1, 1))
    else:
        go = canvas(1, (1, 1))
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

ONE = 1

SEVEN = 7

UNITY = (1, 1)

def equality(
    a: Any,
    b: Any
) -> Boolean:
    """ equality """
    return a == b

def either(
    a: Boolean,
    b: Boolean
) -> Boolean:
    """ logical or """
    return a or b

def branch(
    condition: Boolean,
    if_value: Any,
    else_value: Any
) -> Any:
    """ if else branching """
    return if_value if condition else else_value

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_44f52bb0(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = vmirror(I)
    x1 = equality(x0, I)
    x2 = hmirror(I)
    x3 = equality(x2, I)
    x4 = either(x1, x3)
    x5 = branch(x4, ONE, SEVEN)
    x6 = canvas(x5, UNITY)
    return x6


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("E4", E4_IN, E4_OUT),
        ("E5", E5_IN, E5_OUT),
        ("E6", E6_IN, E6_OUT),
        ("T", T_IN, T_OUT),
        ("T2", T2_IN, T2_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_44f52bb0(inp)
        assert pred == _to_grid(expected), f"{name} failed"

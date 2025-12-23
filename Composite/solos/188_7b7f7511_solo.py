# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "7b7f7511"
SERIAL = "188"
URL    = "https://arcprize.org/play?task=7b7f7511"

# --- Code Golf Concepts ---
CONCEPTS = [
    "separate_images",
    "detect_repetition",
    "crop",
]

# --- Example Grids ---
E1_IN = np.array([
    [1, 1, 3, 2, 1, 1, 3, 2],
    [1, 1, 3, 3, 1, 1, 3, 3],
    [3, 3, 1, 1, 3, 3, 1, 1],
    [2, 3, 1, 1, 2, 3, 1, 1],
], dtype=int)

E1_OUT = np.array([
    [1, 1, 3, 2],
    [1, 1, 3, 3],
    [3, 3, 1, 1],
    [2, 3, 1, 1],
], dtype=int)

E2_IN = np.array([
    [4, 4, 4, 4, 4, 4],
    [6, 4, 8, 6, 4, 8],
    [6, 6, 8, 6, 6, 8],
], dtype=int)

E2_OUT = np.array([
    [4, 4, 4],
    [6, 4, 8],
    [6, 6, 8],
], dtype=int)

E3_IN = np.array([
    [2, 3],
    [3, 2],
    [4, 4],
    [2, 3],
    [3, 2],
    [4, 4],
], dtype=int)

E3_OUT = np.array([
    [2, 3],
    [3, 2],
    [4, 4],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [5, 4, 5],
    [4, 5, 4],
    [6, 6, 4],
    [2, 6, 2],
    [5, 4, 5],
    [4, 5, 4],
    [6, 6, 4],
    [2, 6, 2],
], dtype=int)

T_OUT = np.array([
    [5, 4, 5],
    [4, 5, 4],
    [6, 6, 4],
    [2, 6, 2],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j, A=len):
    return [r[:A(r) // 2] for r in j] if A(j[0]) % 2 < 1 and all((r[:A(r) // 2] == r[A(r) // 2:] for r in j)) else j[:A(j) // 2]


# --- Code Golf Solution (Compressed) ---
def q(g):
    return (X := g[:53 % ~-len(g)]) * (g == X + X) or [*map(p, g)]


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

def generate_7b7f7511(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 15))
    bgc = choice(cols)
    go = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (1, min(9, h * w - 1)))
    colsch = sample(remcols, numc)
    inds = totuple(asindices(go))
    for col in colsch:
        num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = sample(inds, num)
        go = fill(go, col, chos)
        inds = difference(inds, chos)
    if choice((True, False)):
        go = dmirror(go)
        gi = vconcat(go, go)
    else:
        gi = hconcat(go, go)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

def equality(
    a: Any,
    b: Any
) -> Boolean:
    """ equality """
    return a == b

def branch(
    condition: Boolean,
    if_value: Any,
    else_value: Any
) -> Any:
    """ if else branching """
    return if_value if condition else else_value

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

def verify_7b7f7511(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = lefthalf(I)
    x1 = righthalf(I)
    x2 = equality(x0, x1)
    x3 = branch(x2, lefthalf, tophalf)
    x4 = x3(I)
    return x4


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_7b7f7511(inp)
        assert pred == _to_grid(expected), f"{name} failed"

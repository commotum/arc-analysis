# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "9565186b"
SERIAL = "229"
URL    = "https://arcprize.org/play?task=9565186b"

# --- Code Golf Concepts ---
CONCEPTS = [
    "separate_shapes",
    "count_tiles",
    "recoloring",
    "take_maximum",
    "associate_color_to_bools",
]

# --- Example Grids ---
E1_IN = np.array([
    [2, 2, 2],
    [2, 1, 8],
    [2, 8, 8],
], dtype=int)

E1_OUT = np.array([
    [2, 2, 2],
    [2, 5, 5],
    [2, 5, 5],
], dtype=int)

E2_IN = np.array([
    [1, 1, 1],
    [8, 1, 3],
    [8, 2, 2],
], dtype=int)

E2_OUT = np.array([
    [1, 1, 1],
    [5, 1, 5],
    [5, 5, 5],
], dtype=int)

E3_IN = np.array([
    [2, 2, 2],
    [8, 8, 2],
    [2, 2, 2],
], dtype=int)

E3_OUT = np.array([
    [2, 2, 2],
    [5, 5, 2],
    [2, 2, 2],
], dtype=int)

E4_IN = np.array([
    [3, 3, 8],
    [4, 4, 4],
    [8, 1, 1],
], dtype=int)

E4_OUT = np.array([
    [5, 5, 5],
    [4, 4, 4],
    [5, 5, 5],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [1, 3, 2],
    [3, 3, 2],
    [1, 3, 2],
], dtype=int)

T_OUT = np.array([
    [5, 3, 5],
    [3, 3, 5],
    [5, 3, 5],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):A=__import__('collections').Counter([x for R in j for x in R]).most_common(1);c=A[0][0];return[[A if A==c else 5 for A in R]for R in j]


# --- Code Golf Solution (Compressed) ---
def q(m):
    return [[[5, v][v == max((f := sum(m, m)), key=f.count)] for v in r] for r in m]


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

def toindices(
    patch: Patch
) -> Indices:
    """ indices of object cells """
    if len(patch) == 0:
        return frozenset()
    if isinstance(next(iter(patch))[1], tuple):
        return frozenset(index for value, index in patch)
    return patch

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

def generate_9565186b(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(5, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    wg = canvas(5, (h, w))
    numcols = unifint(diff_lb, diff_ub, (2, min(h * w - 1, 8)))
    mostcol = choice(cols)
    nummostcol_lb = (h * w) // numcols + 1
    nummostcol_ub = h * w - numcols + 1
    ubmlb = nummostcol_ub - nummostcol_lb
    nmcdev = unifint(diff_lb, diff_ub, (0, ubmlb))
    nummostcol = nummostcol_ub - nmcdev
    nummostcol = min(max(nummostcol, nummostcol_lb), nummostcol_ub)
    inds = totuple(asindices(wg))
    mostcollocs = sample(inds, nummostcol)
    gi = fill(wg, mostcol, mostcollocs)
    go = fill(wg, mostcol, mostcollocs)
    remcols = remove(mostcol, cols)
    othcols = sample(remcols, numcols - 1)
    reminds = difference(inds, mostcollocs)
    bufferlocs = sample(reminds, numcols - 1)
    for c, l in zip(othcols, bufferlocs):
        gi = fill(gi, c, {l})
    reminds = difference(reminds, bufferlocs)
    colcounts = {c: 1 for c in othcols}
    for ij in reminds:
        if len(othcols) == 0:
            gi = fill(gi, mostcol, {ij})
            go = fill(go, mostcol, {ij})
        else:
            chc = choice(othcols)
            gi = fill(gi, chc, {ij})
            colcounts[chc] += 1
            if colcounts[chc] == nummostcol - 1:
                othcols = remove(chc, othcols)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
IntegerSet = FrozenSet[Integer]

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

FIVE = 5

def size(
    container: Container
) -> Integer:
    """ cardinality """
    return len(container)

def argmax(
    container: Container,
    compfunc: Callable
) -> Any:
    """ largest item by custom order """
    return max(container, key=compfunc, default=None)

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

def partition(
    grid: Grid
) -> Objects:
    """ each cell with the same value part of the same object """
    return frozenset(
        frozenset(
            (v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value
        ) for value in palette(grid)
    )

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

def verify_9565186b(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = shape(I)
    x1 = partition(I)
    x2 = argmax(x1, size)
    x3 = canvas(FIVE, x0)
    x4 = paint(x3, x2)
    return x4


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("E4", E4_IN, E4_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_9565186b(inp)
        assert pred == _to_grid(expected), f"{name} failed"

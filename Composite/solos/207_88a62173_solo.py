# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "88a62173"
SERIAL = "207"
URL    = "https://arcprize.org/play?task=88a62173"

# --- Code Golf Concepts ---
CONCEPTS = [
    "detect_grid",
    "separate_images",
    "find_the_intruder",
    "crop",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 2, 0, 0, 2],
    [2, 2, 0, 2, 2],
    [0, 0, 0, 0, 0],
    [0, 2, 0, 2, 2],
    [2, 2, 0, 2, 0],
], dtype=int)

E1_OUT = np.array([
    [2, 2],
    [2, 0],
], dtype=int)

E2_IN = np.array([
    [1, 0, 0, 1, 0],
    [0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0],
    [1, 0, 0, 1, 0],
    [1, 1, 0, 0, 1],
], dtype=int)

E2_OUT = np.array([
    [1, 0],
    [1, 1],
], dtype=int)

E3_IN = np.array([
    [8, 8, 0, 0, 8],
    [8, 0, 0, 8, 0],
    [0, 0, 0, 0, 0],
    [8, 8, 0, 8, 8],
    [8, 0, 0, 8, 0],
], dtype=int)

E3_OUT = np.array([
    [0, 8],
    [8, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [5, 5, 0, 5, 0],
    [0, 5, 0, 0, 5],
    [0, 0, 0, 0, 0],
    [5, 5, 0, 5, 5],
    [0, 5, 0, 0, 5],
], dtype=int)

T_OUT = np.array([
    [5, 0],
    [0, 5],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
	A={};c=[[[j[0][0],j[0][1]],[j[1][0],j[1][1]]],[[j[3][0],j[3][1]],[j[4][0],j[4][1]]],[[j[0][3],j[0][4]],[j[1][3],j[1][4]]],[[j[3][3],j[3][4]],[j[4][3],j[4][4]]]]
	for E in c:
		E=str(E)
		if E in A:A[E]+=1
		else:A[E]=1
	for E in A:
		if A[E]==1:return eval(E)


# --- Code Golf Solution (Compressed) ---
def q(*g):
    return min(g, key=g.count) if g[3:] else [*map(p, *g, *[h[3:] for h in g])]


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

def generate_88a62173(diff_lb: float, diff_ub: float) -> dict:
    dim_bounds = (1, 30)
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 14))
    w = unifint(diff_lb, diff_ub, (1, 14))
    bgc = choice(cols)
    gib = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (1, min(9, h * w)))
    colsch = sample(remcols, numc)
    inds = totuple(asindices(gib))
    for col in colsch:
        num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = sample(inds, num)
        gib = fill(gib, col, chos)
        inds = difference(inds, chos)
    numchinv = unifint(diff_lb, diff_ub, (0, h * w - 1))
    numch = h * w - numchinv
    inds2 = totuple(asindices(gib))
    subs = sample(inds2, numch)
    go = hmirror(hmirror(gib))
    for x, y in subs:
        go = fill(go, choice(remove(go[x][y], colsch + [bgc])), {(x, y)})
    gi = canvas(bgc, (h*2+1, w*2+1))
    idxes = ((0, 0), (h+1, w+1), (h+1, 0), (0, w+1))
    trgloc = choice(idxes)
    remidxes = remove(trgloc, idxes)
    trgobj = asobject(go)
    otherobj = asobject(gib)
    gi = paint(gi, shift(trgobj, trgloc))
    for ij in remidxes:
        gi = paint(gi, shift(otherobj, ij))
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
def combine(
    a: Container,
    b: Container
) -> Container:
    """ union """
    return type(a)((*a, *b))

def leastcommon(
    container: Container
) -> Any:
    """ least common item """
    return min(set(container), key=container.count)

def astuple(
    a: Integer,
    b: Integer
) -> IntegerTuple:
    """ constructs a tuple """
    return (a, b)

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

def verify_88a62173(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = lefthalf(I)
    x1 = righthalf(I)
    x2 = tophalf(x0)
    x3 = tophalf(x1)
    x4 = bottomhalf(x0)
    x5 = bottomhalf(x1)
    x6 = astuple(x2, x3)
    x7 = astuple(x4, x5)
    x8 = combine(x6, x7)
    x9 = leastcommon(x8)
    return x9


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_88a62173(inp)
        assert pred == _to_grid(expected), f"{name} failed"

# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "77fdfe62"
SERIAL = "183"
URL    = "https://arcprize.org/play?task=77fdfe62"

# --- Code Golf Concepts ---
CONCEPTS = [
    "recoloring",
    "color_guessing",
    "detect_grid",
    "crop",
]

# --- Example Grids ---
E1_IN = np.array([
    [2, 1, 0, 0, 0, 0, 1, 3],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 0, 8, 0, 0, 1, 0],
    [0, 1, 8, 8, 0, 8, 1, 0],
    [0, 1, 0, 0, 8, 0, 1, 0],
    [0, 1, 8, 0, 8, 8, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [4, 1, 0, 0, 0, 0, 1, 6],
], dtype=int)

E1_OUT = np.array([
    [0, 2, 0, 0],
    [2, 2, 0, 3],
    [0, 0, 6, 0],
    [4, 0, 6, 6],
], dtype=int)

E2_IN = np.array([
    [9, 1, 0, 0, 1, 4],
    [1, 1, 1, 1, 1, 1],
    [0, 1, 8, 8, 1, 0],
    [0, 1, 8, 0, 1, 0],
    [1, 1, 1, 1, 1, 1],
    [2, 1, 0, 0, 1, 3],
], dtype=int)

E2_OUT = np.array([
    [9, 4],
    [2, 0],
], dtype=int)

E3_IN = np.array([
    [6, 1, 0, 0, 0, 0, 1, 2],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 0, 8, 0, 8, 1, 0],
    [0, 1, 8, 8, 8, 0, 1, 0],
    [0, 1, 8, 0, 8, 8, 1, 0],
    [0, 1, 8, 8, 8, 0, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [7, 1, 0, 0, 0, 0, 1, 4],
], dtype=int)

E3_OUT = np.array([
    [0, 6, 0, 2],
    [6, 6, 2, 0],
    [7, 0, 4, 4],
    [7, 7, 4, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [3, 1, 0, 0, 0, 0, 0, 0, 1, 4],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 0, 8, 8, 0, 0, 0, 1, 0],
    [0, 1, 8, 8, 8, 0, 8, 0, 1, 0],
    [0, 1, 0, 0, 8, 0, 8, 0, 1, 0],
    [0, 1, 0, 8, 0, 8, 8, 0, 1, 0],
    [0, 1, 8, 8, 0, 8, 0, 8, 1, 0],
    [0, 1, 0, 8, 0, 0, 8, 0, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [7, 1, 0, 0, 0, 0, 0, 0, 1, 5],
], dtype=int)

T_OUT = np.array([
    [0, 3, 3, 0, 0, 0],
    [3, 3, 3, 0, 4, 0],
    [0, 0, 3, 0, 4, 0],
    [0, 7, 0, 5, 5, 0],
    [7, 7, 0, 5, 0, 5],
    [0, 7, 0, 0, 5, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
	A=range;c=len(j);E=c//2-2;k=[];W=[j[0][0],j[0][-1],j[-1][0],j[-1][-1]]
	for l in A(2,c-2):
		J=[]
		for a in A(2,c-2):
			C=j[l][a]
			if C==8:e=(l-2)//E;K=(a-2)//E;C=W[e*2+K]
			J.append(C)
		k.append(J)
	return k


# --- Code Golf Solution (Compressed) ---
def q(g, h=0):
    return g * 0 != 0 and [*map(p, len(g) // 2 * g[:1] + g[-1:] * 9, h or g)][2:-2] or h % 7 * g


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import sample, uniform

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

def totuple(
    container: FrozenSet
) -> Tuple:
    """ conversion to tuple """
    return tuple(container)

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

def generate_77fdfe62(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 13))
    w = unifint(diff_lb, diff_ub, (1, 13))
    c1, c2, c3, c4, barc, bgc, inc = sample(cols, 7)
    qd = canvas(bgc, (h, w))
    inds = totuple(asindices(qd))
    fullh = 2 * h + 4
    fullw = 2 * w + 4
    n1 = unifint(diff_lb, diff_ub, (1, h * w))
    n2 = unifint(diff_lb, diff_ub, (1, h * w))
    n3 = unifint(diff_lb, diff_ub, (1, h * w))
    n4 = unifint(diff_lb, diff_ub, (1, h * w))
    i1 = sample(inds, n1)
    i2 = sample(inds, n2)
    i3 = sample(inds, n3)
    i4 = sample(inds, n4)
    gi = canvas(bgc, (2 * h + 4, 2 * w + 4))
    gi = fill(gi, barc, connect((1, 0), (1, fullw - 1)))
    gi = fill(gi, barc, connect((fullh - 2, 0), (fullh - 2, fullw - 1)))
    gi = fill(gi, barc, connect((0, 1), (fullh - 1, 1)))
    gi = fill(gi, barc, connect((0, fullw - 2), (fullh - 1, fullw - 2)))
    gi = fill(gi, c1, {(0, 0)})
    gi = fill(gi, c2, {(0, fullw - 1)})
    gi = fill(gi, c3, {(fullh - 1, 0)})
    gi = fill(gi, c4, {(fullh - 1, fullw - 1)})
    gi = fill(gi, inc, shift(i1, (2, 2)))
    gi = fill(gi, inc, shift(i2, (2, 2+w)))
    gi = fill(gi, inc, shift(i3, (2+h, 2)))
    gi = fill(gi, inc, shift(i4, (2+h, 2+w)))
    go = canvas(bgc, (2 * h, 2 * w))
    go = fill(go, c1, shift(i1, (0, 0)))
    go = fill(go, c2, shift(i2, (0, w)))
    go = fill(go, c3, shift(i3, (h, 0)))
    go = fill(go, c4, shift(i4, (h, w)))
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Numerical = Union[Integer, IntegerTuple]

IntegerSet = FrozenSet[Integer]

Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

ORIGIN = (0, 0)

def difference(
    a: Container,
    b: Container
) -> Container:
    """ difference """
    return type(a)(e for e in a if e not in b)

def decrement(
    x: Numerical
) -> Numerical:
    """ decrementing """
    return x - 1 if isinstance(x, int) else (x[0] - 1, x[1] - 1)

def toivec(
    i: Integer
) -> IntegerTuple:
    """ vector pointing vertically """
    return (i, 0)

def tojvec(
    j: Integer
) -> IntegerTuple:
    """ vector pointing horizontally """
    return (0, j)

def first(
    container: Container
) -> Any:
    """ first item of container """
    return next(iter(container))

def remove(
    value: Any,
    container: Container
) -> Container:
    """ remove item from container """
    return type(container)(e for e in container if e != value)

def other(
    container: Container,
    value: Any
) -> Any:
    """ other value in the container """
    return first(remove(value, container))

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

def ulcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of upper left corner """
    return tuple(map(min, zip(*toindices(patch))))

def urcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of upper right corner """
    return tuple(map(lambda ix: {0: min, 1: max}[ix[0]](ix[1]), enumerate(zip(*toindices(patch)))))

def llcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of lower left corner """
    return tuple(map(lambda ix: {0: max, 1: min}[ix[0]](ix[1]), enumerate(zip(*toindices(patch)))))

def lrcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of lower right corner """
    return tuple(map(max, zip(*toindices(patch))))

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

def corners(
    patch: Patch
) -> Indices:
    """ indices of corners """
    return frozenset({ulcorner(patch), urcorner(patch), llcorner(patch), lrcorner(patch)})

def trim(
    grid: Grid
) -> Grid:
    """ trim border of grid """
    return tuple(r[1:-1] for r in grid[1:-1])

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

def box(
    patch: Patch
) -> Indices:
    """ outline of patch """
    if len(patch) == 0:
        return patch
    ai, aj = ulcorner(patch)
    bi, bj = lrcorner(patch)
    si, sj = min(ai, bi), min(aj, bj)
    ei, ej = max(ai, bi), max(aj, bj)
    vlines = {(i, sj) for i in range(si, ei + 1)} | {(i, ej) for i in range(si, ei + 1)}
    hlines = {(si, j) for j in range(sj, ej + 1)} | {(ei, j) for j in range(sj, ej + 1)}
    return frozenset(vlines | hlines)

def compress(
    grid: Grid
) -> Grid:
    """ removes frontiers from grid """
    ri = tuple(i for i, r in enumerate(grid) if len(set(r)) == 1)
    ci = tuple(j for j, c in enumerate(dmirror(grid)) if len(set(c)) == 1)
    return tuple(tuple(v for j, v in enumerate(r) if j not in ci) for i, r in enumerate(grid) if i not in ri)

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_77fdfe62(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = trim(I)
    x1 = trim(x0)
    x2 = tophalf(x1)
    x3 = lefthalf(x2)
    x4 = tophalf(x1)
    x5 = righthalf(x4)
    x6 = bottomhalf(x1)
    x7 = lefthalf(x6)
    x8 = bottomhalf(x1)
    x9 = righthalf(x8)
    x10 = index(I, ORIGIN)
    x11 = width(I)
    x12 = decrement(x11)
    x13 = tojvec(x12)
    x14 = index(I, x13)
    x15 = height(I)
    x16 = decrement(x15)
    x17 = toivec(x16)
    x18 = index(I, x17)
    x19 = shape(I)
    x20 = decrement(x19)
    x21 = index(I, x20)
    x22 = compress(I)
    x23 = asindices(x22)
    x24 = box(x23)
    x25 = corners(x23)
    x26 = difference(x24, x25)
    x27 = toobject(x26, x22)
    x28 = color(x27)
    x29 = palette(x1)
    x30 = other(x29, x28)
    x31 = replace(x3, x30, x10)
    x32 = replace(x5, x30, x14)
    x33 = replace(x7, x30, x18)
    x34 = replace(x9, x30, x21)
    x35 = hconcat(x31, x32)
    x36 = hconcat(x33, x34)
    x37 = vconcat(x35, x36)
    return x37


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_77fdfe62(inp)
        assert pred == _to_grid(expected), f"{name} failed"

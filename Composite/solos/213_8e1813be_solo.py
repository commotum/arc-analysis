# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "8e1813be"
SERIAL = "213"
URL    = "https://arcprize.org/play?task=8e1813be"

# --- Code Golf Concepts ---
CONCEPTS = [
    "recoloring",
    "color_guessing",
    "direction_guesingcrop",
    "image_within_image",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 5, 5, 5, 5, 5, 5, 0, 4, 4, 4, 4, 4, 4, 4],
    [0, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 5, 5, 5, 5, 5, 5, 0, 8, 8, 8, 8, 8, 8, 8],
    [0, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
], dtype=int)

E1_OUT = np.array([
    [2, 2, 2, 2, 2, 2],
    [1, 1, 1, 1, 1, 1],
    [3, 3, 3, 3, 3, 3],
    [4, 4, 4, 4, 4, 4],
    [8, 8, 8, 8, 8, 8],
    [6, 6, 6, 6, 6, 6],
], dtype=int)

E2_IN = np.array([
    [0, 0, 0, 0, 0, 2, 0, 0, 4, 0],
    [0, 5, 5, 5, 0, 2, 0, 0, 4, 0],
    [0, 5, 5, 5, 0, 2, 0, 0, 4, 0],
    [0, 5, 5, 5, 0, 2, 0, 0, 4, 0],
    [0, 0, 0, 0, 0, 2, 0, 0, 4, 0],
    [0, 0, 1, 0, 0, 2, 0, 0, 4, 0],
    [0, 0, 1, 0, 0, 2, 0, 0, 4, 0],
    [0, 0, 1, 0, 0, 2, 0, 0, 4, 0],
    [0, 0, 1, 0, 0, 2, 0, 0, 4, 0],
    [0, 0, 1, 0, 0, 2, 0, 0, 4, 0],
    [0, 0, 1, 0, 0, 2, 0, 0, 4, 0],
    [0, 0, 1, 0, 0, 2, 0, 0, 4, 0],
], dtype=int)

E2_OUT = np.array([
    [1, 2, 4],
    [1, 2, 4],
    [1, 2, 4],
], dtype=int)

E3_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 2, 2, 2, 0, 5, 5, 5, 5, 0, 2],
    [0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 0, 0],
    [0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 0, 0],
    [8, 8, 8, 8, 8, 0, 5, 5, 5, 5, 0, 8],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [2, 2, 2, 2],
    [8, 8, 8, 8],
    [4, 4, 4, 4],
    [1, 1, 1, 1],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0],
    [3, 3, 3, 3, 3, 3, 3, 0, 5, 5, 5, 5, 5, 5, 5, 0, 3, 3, 3],
    [0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 0, 5, 5, 5, 5, 5, 5, 5, 0, 8, 8, 8],
    [0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
], dtype=int)

T_OUT = np.array([
    [2, 2, 2, 2, 2, 2, 2],
    [3, 3, 3, 3, 3, 3, 3],
    [8, 8, 8, 8, 8, 8, 8],
    [4, 4, 4, 4, 4, 4, 4],
    [6, 6, 6, 6, 6, 6, 6],
    [1, 1, 1, 1, 1, 1, 1],
    [7, 7, 7, 7, 7, 7, 7],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def Z(j,A):return len(set([J[A]for J in j]))
def p(c):
	E=enumerate;k,W=len(c),len(c[0]);l=Z(c,0)+Z(c,-1)<len(set(c[0]))+len(set(c[-1]));c=[[J if J!=5 else 0 for J in J]for J in c]
	for(J,a)in E(c):
		for(C,e)in E(a):
			if l:c[J][C]=max([c[0][C],c[-1][C]])
			else:c[J][C]=max([c[J][0],c[J][-1]])
	if l:c=[[J for J in J if J>0]for J in c];c=c[:len(c[0])]
	else:c=[J for J in c if sum(J)>0];c=[J[:len(c)]for J in c]
	return c


# --- Code Golf Solution (Compressed) ---
def q(g):
    return [l[o:] for r in g if (l := [e for e in r if e % 5])[~(o := (6 - len({*'%s' % g}))):]][o:]


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, randint, sample, uniform

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

Piece = Union[Grid, Patch]

def repeat(
    item: Any,
    num: Integer
) -> Tuple:
    """ repetition of item within vector """
    return tuple(item for i in range(num))

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

def backdrop(
    patch: Patch
) -> Indices:
    """ indices in bounding box of patch """
    if len(patch) == 0:
        return frozenset({})
    indices = toindices(patch)
    si, sj = ulcorner(indices)
    ei, ej = lrcorner(patch)
    return frozenset((i, j) for i in range(si, ei + 1) for j in range(sj, ej + 1))

def outbox(
    patch: Patch
) -> Indices:
    """ outbox for patch """
    ai, aj = uppermost(patch) - 1, leftmost(patch) - 1
    bi, bj = lowermost(patch) + 1, rightmost(patch) + 1
    si, sj = min(ai, bi), min(aj, bj)
    ei, ej = max(ai, bi), max(aj, bj)
    vlines = {(i, sj) for i in range(si, ei + 1)} | {(i, ej) for i in range(si, ei + 1)}
    hlines = {(si, j) for j in range(sj, ej + 1)} | {(ei, j) for j in range(sj, ej + 1)}
    return frozenset(vlines | hlines)

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

def generate_8e1813be(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    bgc, sqc = sample(cols, 2)
    remcols = remove(bgc, remove(sqc, cols))
    nbars = unifint(diff_lb, diff_ub, (3, 8))
    ccols = sample(remcols, nbars)
    w = unifint(diff_lb, diff_ub, (nbars+3, 30))
    hmarg = unifint(diff_lb, diff_ub, (2 * nbars, 30 - nbars))
    ccols = list(ccols)
    go = tuple(repeat(col, nbars) for col in ccols)
    gi = tuple(repeat(col, w) for col in ccols)
    r = repeat(bgc, w)
    for k in range(hmarg):
        idx = randint(0, len(go) - 1)
        gi = gi[:idx] + (r,) + gi[idx:]
    h2 = nbars + hmarg
    oh, ow = nbars, nbars
    loci = randint(1, h2 - oh - 2)
    locj = randint(1, w - ow - 2)
    sq = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
    gi = fill(gi, sqc, sq)
    gi = fill(gi, bgc, outbox(sq))
    if choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

IntegerSet = FrozenSet[Integer]

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

ONE = 1

def identity(
    x: Any
) -> Any:
    """ identity function """
    return x

def order(
    container: Container,
    compfunc: Callable
) -> Tuple:
    """ order container by custom key """
    return tuple(sorted(container, key=compfunc))

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

def either(
    a: Boolean,
    b: Boolean
) -> Boolean:
    """ logical or """
    return a or b

def sfilter(
    container: Container,
    condition: Callable
) -> Container:
    """ keep elements in container that satisfy condition """
    return type(container)(e for e in container if condition(e))

def branch(
    condition: Boolean,
    if_value: Any,
    else_value: Any
) -> Any:
    """ if else branching """
    return if_value if condition else else_value

def matcher(
    function: Callable,
    target: Any
) -> Callable:
    """ construction of equality function """
    return lambda x: function(x) == target

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

def partition(
    grid: Grid
) -> Objects:
    """ each cell with the same value part of the same object """
    return frozenset(
        frozenset(
            (v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value
        ) for value in palette(grid)
    )

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

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_8e1813be(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = partition(I)
    x1 = matcher(height, ONE)
    x2 = matcher(width, ONE)
    x3 = fork(either, x1, x2)
    x4 = sfilter(x0, x3)
    x5 = matcher(height, ONE)
    x6 = sfilter(x4, x5)
    x7 = size(x6)
    x8 = matcher(width, ONE)
    x9 = sfilter(x4, x8)
    x10 = size(x9)
    x11 = greater(x7, x10)
    x12 = branch(x11, dmirror, identity)
    x13 = branch(x11, uppermost, leftmost)
    x14 = order(x4, x13)
    x15 = apply(color, x14)
    x16 = size(x4)
    x17 = repeat(x15, x16)
    x18 = x12(x17)
    return x18


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_8e1813be(inp)
        assert pred == _to_grid(expected), f"{name} failed"

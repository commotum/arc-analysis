# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "363442ee"
SERIAL = "075"
URL    = "https://arcprize.org/play?task=363442ee"

# --- Code Golf Concepts ---
CONCEPTS = [
    "detect_wall",
    "pattern_repetition",
    "pattern_juxtaposition",
]

# --- Example Grids ---
E1_IN = np.array([
    [4, 2, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 6, 2, 5, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [6, 4, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [4, 2, 2, 5, 4, 2, 2, 0, 0, 0, 0, 0, 0],
    [2, 6, 2, 5, 2, 6, 2, 0, 0, 0, 0, 0, 0],
    [6, 4, 4, 5, 6, 4, 4, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 4, 2, 2, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 2, 6, 2, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 6, 4, 4, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 4, 2, 2, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 2, 6, 2, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 6, 4, 4, 0, 0, 0],
], dtype=int)

E2_IN = np.array([
    [2, 7, 3, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 3, 3, 5, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [3, 7, 7, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 5, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 5, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [2, 7, 3, 5, 0, 0, 0, 2, 7, 3, 0, 0, 0],
    [2, 3, 3, 5, 0, 0, 0, 2, 3, 3, 0, 0, 0],
    [3, 7, 7, 5, 0, 0, 0, 3, 7, 7, 0, 0, 0],
    [0, 0, 0, 5, 2, 7, 3, 0, 0, 0, 2, 7, 3],
    [0, 0, 0, 5, 2, 3, 3, 0, 0, 0, 2, 3, 3],
    [0, 0, 0, 5, 3, 7, 7, 0, 0, 0, 3, 7, 7],
    [0, 0, 0, 5, 2, 7, 3, 2, 7, 3, 0, 0, 0],
    [0, 0, 0, 5, 2, 3, 3, 2, 3, 3, 0, 0, 0],
    [0, 0, 0, 5, 3, 7, 7, 3, 7, 7, 0, 0, 0],
], dtype=int)

E3_IN = np.array([
    [3, 8, 6, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [9, 8, 2, 5, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    [9, 9, 9, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [3, 8, 6, 5, 3, 8, 6, 0, 0, 0, 3, 8, 6],
    [9, 8, 2, 5, 9, 8, 2, 0, 0, 0, 9, 8, 2],
    [9, 9, 9, 5, 9, 9, 9, 0, 0, 0, 9, 9, 9],
    [0, 0, 0, 5, 0, 0, 0, 3, 8, 6, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 9, 8, 2, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 9, 9, 9, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 3, 8, 6, 3, 8, 6],
    [0, 0, 0, 5, 0, 0, 0, 9, 8, 2, 9, 8, 2],
    [0, 0, 0, 5, 0, 0, 0, 9, 9, 9, 9, 9, 9],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [3, 3, 9, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 4, 4, 5, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    [8, 9, 8, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 5, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 5, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

T_OUT = np.array([
    [3, 3, 9, 5, 0, 0, 0, 3, 3, 9, 3, 3, 9],
    [8, 4, 4, 5, 0, 0, 0, 8, 4, 4, 8, 4, 4],
    [8, 9, 8, 5, 0, 0, 0, 8, 9, 8, 8, 9, 8],
    [0, 0, 0, 5, 3, 3, 9, 0, 0, 0, 3, 3, 9],
    [0, 0, 0, 5, 8, 4, 4, 0, 0, 0, 8, 4, 4],
    [0, 0, 0, 5, 8, 9, 8, 0, 0, 0, 8, 9, 8],
    [0, 0, 0, 5, 3, 3, 9, 3, 3, 9, 0, 0, 0],
    [0, 0, 0, 5, 8, 4, 4, 8, 4, 4, 0, 0, 0],
    [0, 0, 0, 5, 8, 9, 8, 8, 9, 8, 0, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
	A=range;c=[A[:]for A in j];E=[[j[k][A]for A in A(3)]for k in A(3)]
	for k in A(9):
		for W in A(4,13):
			if j[k][W]==1:
				for l in A(-1,2):
					for J in A(-1,2):
						if 0<=k+l<9 and 4<=W+J<13:c[k+l][W+J]=E[l+1][J+1]
	return c


# --- Code Golf Solution (Compressed) ---
def q(m):
    return [m[y][:4] + [m[y - y % 3 + 1][x - x % 3 + 5] * m[y % 3][x % 3] for x in R] for y in R]


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

Piece = Union[Grid, Patch]

def identity(
    x: Any
) -> Any:
    """ identity function """
    return x

def add(
    a: Numerical,
    b: Numerical
) -> Numerical:
    """ addition """
    if isinstance(a, int) and isinstance(b, int):
        return a + b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] + b[0], a[1] + b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a + b[0], a + b[1])
    return (a[0] + b, a[1] + b)

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

def product(
    a: Container,
    b: Container
) -> FrozenSet:
    """ cartesian product """
    return frozenset((i, j) for j in b for i in a)

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

def generate_363442ee(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 3))
    w = unifint(diff_lb, diff_ub, (1, 3))
    h = h * 2 + 1
    w = w * 2 + 1
    nremh = unifint(diff_lb, diff_ub, (2, 30 // h))
    nremw = unifint(diff_lb, diff_ub, (2, (30 - w - 1) // w))
    rsh = nremh * h
    rsw = nremw * w
    rss = (rsh, rsw)
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    barcol = choice(remcols)
    remcols = remove(barcol, remcols)
    rsi = canvas(bgc, rss)
    rso = canvas(bgc, rss)
    ls = canvas(bgc, ((nremh - 1) * h, w))
    ulc = canvas(bgc, (h, w))
    bar = canvas(barcol, (nremh * h, 1))
    dotcands = totuple(product(interval(0, rsh, h), interval(0, rsw, w)))
    dotcol = choice(remcols)
    dev = unifint(diff_lb, diff_ub, (1, len(dotcands) // 2))
    ndots = choice((dev, len(dotcands) - dev))
    ndots = min(max(1, ndots), len(dotcands))
    dots = sample(dotcands, ndots)
    nfullremcols = unifint(diff_lb, diff_ub, (1, 8))
    fullremcols = sample(remcols, nfullremcols)
    for ij in asindices(ulc):
        ulc = fill(ulc, choice(fullremcols), {ij})
    ulco = asobject(ulc)
    osf = (h//2, w//2)
    for d in dots:
        rsi = fill(rsi, dotcol, {add(osf, d)})
        rso = paint(rso, shift(ulco, d))
    gi = hconcat(hconcat(vconcat(ulc, ls), bar), rsi)
    go = hconcat(hconcat(vconcat(ulc, ls), bar), rso)
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

ContainerContainer = Container[Container]

F = False

T = True

def invert(
    n: Numerical
) -> Numerical:
    """ inversion with respect to addition """
    return -n if isinstance(n, int) else (-n[0], -n[1])

def halve(
    n: Numerical
) -> Numerical:
    """ scaling by one half """
    return n // 2 if isinstance(n, int) else (n[0] // 2, n[1] // 2)

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

def argmax(
    container: Container,
    compfunc: Callable
) -> Any:
    """ largest item by custom order """
    return max(container, key=compfunc, default=None)

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

def normalize(
    patch: Patch
) -> Patch:
    """ moves upper left corner to origin """
    if len(patch) == 0:
        return patch
    return shift(patch, (-uppermost(patch), -leftmost(patch)))

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

def center(
    patch: Patch
) -> IntegerTuple:
    """ center of the patch """
    return (uppermost(patch) + height(patch) // 2, leftmost(patch) + width(patch) // 2)

def frontiers(
    grid: Grid
) -> Objects:
    """ set of frontiers """
    h, w = len(grid), len(grid[0])
    row_indices = tuple(i for i, r in enumerate(grid) if len(set(r)) == 1)
    column_indices = tuple(j for j, c in enumerate(dmirror(grid)) if len(set(c)) == 1)
    hfrontiers = frozenset({frozenset({(grid[i][j], (i, j)) for j in range(w)}) for i in row_indices})
    vfrontiers = frozenset({frozenset({(grid[i][j], (i, j)) for i in range(h)}) for j in column_indices})
    return hfrontiers | vfrontiers

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_363442ee(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = frontiers(I)
    x1 = merge(x0)
    x2 = mostcolor(I)
    x3 = fill(I, x2, x1)
    x4 = objects(x3, F, F, T)
    x5 = argmax(x4, size)
    x6 = remove(x5, x4)
    x7 = apply(center, x6)
    x8 = normalize(x5)
    x9 = shape(x5)
    x10 = halve(x9)
    x11 = invert(x10)
    x12 = shift(x8, x11)
    x13 = lbind(shift, x12)
    x14 = mapply(x13, x7)
    x15 = paint(I, x14)
    return x15


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_363442ee(inp)
        assert pred == _to_grid(expected), f"{name} failed"

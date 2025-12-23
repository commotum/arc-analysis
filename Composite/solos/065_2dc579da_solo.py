# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "2dc579da"
SERIAL = "065"
URL    = "https://arcprize.org/play?task=2dc579da"

# --- Code Golf Concepts ---
CONCEPTS = [
    "detect_grid",
    "find_the_intruder",
    "crop",
]

# --- Example Grids ---
E1_IN = np.array([
    [8, 8, 3, 8, 8],
    [8, 8, 3, 8, 8],
    [3, 3, 3, 3, 3],
    [8, 8, 3, 8, 8],
    [4, 8, 3, 8, 8],
], dtype=int)

E1_OUT = np.array([
    [8, 8],
    [4, 8],
], dtype=int)

E2_IN = np.array([
    [4, 4, 4, 2, 4, 4, 4],
    [4, 4, 4, 2, 4, 1, 4],
    [4, 4, 4, 2, 4, 4, 4],
    [2, 2, 2, 2, 2, 2, 2],
    [4, 4, 4, 2, 4, 4, 4],
    [4, 4, 4, 2, 4, 4, 4],
    [4, 4, 4, 2, 4, 4, 4],
], dtype=int)

E2_OUT = np.array([
    [4, 4, 4],
    [4, 1, 4],
    [4, 4, 4],
], dtype=int)

E3_IN = np.array([
    [3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3],
    [3, 8, 3, 3, 3, 1, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3],
], dtype=int)

E3_OUT = np.array([
    [3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3],
    [3, 8, 3, 3, 3],
    [3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 1, 2, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
], dtype=int)

T_OUT = np.array([
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
    [1, 2, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
	A=range;c=(len(j)-1)//2
	if c==1:
		E=[j[0][0],j[0][2],j[2][0],j[2][2]]
		for k in E:
			if E.count(k)==1:return[[k]]
	for(W,l)in[(0,0),(0,c+1),(c+1,0),(c+1,c+1)]:
		J=[[j[W+k][l+c]for c in A(c)]for k in A(c)];k=[J[k][E]for k in A(c)for E in A(c)]
		if len(set(k))>1:return J


# --- Code Golf Solution (Compressed) ---
def q(*g):
    return min(g, key=g.count) if g[3:] else [*map(p, *g, *[h[len(h) // 2 + 1:] for h in g])]


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, randint, uniform

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

def identity(
    x: Any
) -> Any:
    """ identity function """
    return x

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

def crop(
    grid: Grid,
    start: IntegerTuple,
    dims: IntegerTuple
) -> Grid:
    """ subgrid specified by start and dimension """
    return tuple(r[start[1]:start[1]+dims[1]] for r in grid[start[0]:start[0]+dims[0]])

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

def generate_2dc579da(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    linc = choice(remcols)
    remcols = remove(linc, remcols)
    dotc = choice(remcols)
    hdev = unifint(diff_lb, diff_ub, (0, (h - 2) // 2))
    lineh = choice((hdev, h - 2 - hdev))
    lineh = max(min(h - 2, lineh), 1)
    wdev = unifint(diff_lb, diff_ub, (0, (w - 2) // 2))
    linew = choice((wdev, w - 2 - wdev))
    linew = max(min(w - 2, linew), 1)
    locidev = unifint(diff_lb, diff_ub, (1, h // 2))
    loci = choice((h // 2 - locidev, h // 2 + locidev))
    loci = min(max(1, loci), h - lineh - 1)
    locjdev = unifint(diff_lb, diff_ub, (1, w // 2))
    locj = choice((w // 2 - locjdev, w // 2 + locjdev))
    locj = min(max(1, locj), w - linew - 1)
    gi = canvas(bgc, (h, w))
    for a in range(loci, loci + lineh):
        gi = fill(gi, linc, connect((a, 0), (a, w - 1)))
    for b in range(locj, locj + linew):
        gi = fill(gi, linc, connect((0, b), (h - 1, b)))
    doth = randint(1, loci)
    dotw = randint(1, locj)
    dotloci = randint(0, loci - doth)
    dotlocj = randint(0, locj - dotw)
    dot = backdrop(frozenset({(dotloci, dotlocj), (dotloci + doth - 1, dotlocj + dotw - 1)}))
    gi = fill(gi, dotc, dot)
    go = crop(gi, (0, 0), (loci, locj))
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Numerical = Union[Integer, IntegerTuple]

IntegerSet = FrozenSet[Integer]

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

ContainerContainer = Container[Container]

ONE = 1

ORIGIN = (0, 0)

UNITY = (1, 1)

NEG_UNITY = (-1, -1)

UP_RIGHT = (-1, 1)

DOWN_LEFT = (1, -1)

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

def flip(
    b: Boolean
) -> Boolean:
    """ logical not """
    return not b

def combine(
    a: Container,
    b: Container
) -> Container:
    """ union """
    return type(a)((*a, *b))

def merge(
    containers: ContainerContainer
) -> Container:
    """ merging """
    return type(containers)(e for c in containers for e in c)

def mostcommon(
    container: Container
) -> Any:
    """ most common item """
    return max(set(container), key=container.count)

def initset(
    value: Any
) -> FrozenSet:
    """ initialize container """
    return frozenset({value})

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

def sfilter(
    container: Container,
    condition: Callable
) -> Container:
    """ keep elements in container that satisfy condition """
    return type(container)(e for e in container if condition(e))

def mfilter(
    container: Container,
    function: Callable
) -> FrozenSet:
    """ filter and merge """
    return merge(sfilter(container, function))

def extract(
    container: Container,
    condition: Callable
) -> Any:
    """ first element of container that satisfies condition """
    return next(e for e in container if condition(e))

def insert(
    value: Any,
    container: FrozenSet
) -> FrozenSet:
    """ insert item into container """
    return container.union(frozenset({value}))

def astuple(
    a: Integer,
    b: Integer
) -> IntegerTuple:
    """ constructs a tuple """
    return (a, b)

def compose(
    outer: Callable,
    inner: Callable
) -> Callable:
    """ function composition """
    return lambda x: outer(inner(x))

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

def vline(
    patch: Patch
) -> Boolean:
    """ whether the piece forms a vertical line """
    return height(patch) == len(patch) and width(patch) == 1

def hline(
    patch: Patch
) -> Boolean:
    """ whether the piece forms a horizontal line """
    return width(patch) == len(patch) and height(patch) == 1

def palette(
    element: Element
) -> IntegerSet:
    """ colors occurring in object or grid """
    if isinstance(element, tuple):
        return frozenset({v for r in element for v in r})
    return frozenset({v for v, _ in element})

def numcolors(
    element: Element
) -> IntegerSet:
    """ number of colors occurring in object or grid """
    return len(palette(element))

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

def subgrid(
    patch: Patch,
    grid: Grid
) -> Grid:
    """ smallest subgrid containing object """
    return crop(grid, ulcorner(patch), shape(patch))

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

def verify_2dc579da(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = frontiers(I)
    x1 = mfilter(x0, hline)
    x2 = mfilter(x0, vline)
    x3 = uppermost(x1)
    x4 = leftmost(x2)
    x5 = astuple(x3, x4)
    x6 = add(x5, NEG_UNITY)
    x7 = uppermost(x1)
    x8 = rightmost(x2)
    x9 = astuple(x7, x8)
    x10 = add(x9, UP_RIGHT)
    x11 = lowermost(x1)
    x12 = leftmost(x2)
    x13 = astuple(x11, x12)
    x14 = add(x13, DOWN_LEFT)
    x15 = lowermost(x1)
    x16 = rightmost(x2)
    x17 = astuple(x15, x16)
    x18 = add(x17, UNITY)
    x19 = initset(ORIGIN)
    x20 = insert(x6, x19)
    x21 = width(I)
    x22 = decrement(x21)
    x23 = tojvec(x22)
    x24 = initset(x23)
    x25 = insert(x10, x24)
    x26 = height(I)
    x27 = decrement(x26)
    x28 = toivec(x27)
    x29 = initset(x28)
    x30 = insert(x14, x29)
    x31 = shape(I)
    x32 = decrement(x31)
    x33 = initset(x32)
    x34 = insert(x18, x33)
    x35 = astuple(x20, x25)
    x36 = astuple(x30, x34)
    x37 = combine(x35, x36)
    x38 = rbind(toobject, I)
    x39 = compose(x38, backdrop)
    x40 = apply(x39, x37)
    x41 = matcher(numcolors, ONE)
    x42 = sfilter(x40, x41)
    x43 = apply(color, x42)
    x44 = mostcommon(x43)
    x45 = initset(x44)
    x46 = matcher(palette, x45)
    x47 = compose(flip, x46)
    x48 = extract(x40, x47)
    x49 = subgrid(x48, I)
    return x49


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_2dc579da(inp)
        assert pred == _to_grid(expected), f"{name} failed"

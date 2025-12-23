# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "fcb5c309"
SERIAL = "396"
URL    = "https://arcprize.org/play?task=fcb5c309"

# --- Code Golf Concepts ---
CONCEPTS = [
    "rectangle_guessing",
    "separate_images",
    "count_tiles",
    "take_maximum",
    "crop",
    "recoloring",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 4],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0],
    [2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 2, 2],
    [2, 4, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 2],
    [2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 2],
    [2, 0, 0, 0, 4, 0, 2, 0, 0, 0, 2, 2, 2],
    [2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
    [2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 4, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 4, 0],
], dtype=int)

E1_OUT = np.array([
    [4, 4, 4, 4, 4, 4, 4],
    [4, 0, 0, 0, 0, 0, 4],
    [4, 4, 0, 0, 0, 0, 4],
    [4, 0, 0, 0, 0, 0, 4],
    [4, 0, 0, 0, 4, 0, 4],
    [4, 0, 0, 0, 0, 0, 4],
    [4, 4, 4, 4, 4, 4, 4],
], dtype=int)

E2_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0],
    [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 3, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 3, 0, 0, 1, 0],
    [0, 3, 0, 0, 0, 3, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 3, 0, 1, 3, 0, 3, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 3, 0, 0, 0, 0],
    [3, 0, 0, 0, 0, 0, 0, 3, 0, 0, 3, 3, 0, 0, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [3, 3, 3, 3, 3, 3, 3],
    [3, 0, 0, 3, 0, 0, 3],
    [3, 0, 0, 0, 0, 0, 3],
    [3, 3, 0, 3, 0, 0, 3],
    [3, 0, 0, 0, 0, 0, 3],
    [3, 3, 3, 3, 3, 3, 3],
], dtype=int)

E3_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
    [0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 2, 0, 0, 0, 0, 0],
    [0, 3, 0, 2, 0, 2, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 3, 0, 0, 0, 0, 0, 3, 0, 0, 3, 3, 3, 3, 0, 0],
    [0, 3, 0, 0, 0, 0, 0, 3, 2, 0, 3, 0, 2, 3, 0, 0],
    [0, 3, 0, 2, 0, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0],
    [0, 3, 0, 0, 0, 0, 2, 3, 0, 0, 3, 0, 0, 3, 0, 0],
    [0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 3, 3, 3, 3, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
    [0, 0, 0, 3, 3, 3, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2],
    [0, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
    [0, 0, 2, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [2, 2, 2, 2, 2, 2, 2],
    [2, 0, 2, 0, 2, 0, 2],
    [2, 0, 0, 0, 0, 0, 2],
    [2, 0, 0, 0, 0, 0, 2],
    [2, 0, 2, 0, 0, 0, 2],
    [2, 0, 0, 0, 0, 2, 2],
    [2, 2, 2, 2, 2, 2, 2],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 1, 0, 8, 0, 8, 0, 0, 1, 8, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 8, 0, 8, 0, 0],
    [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 8, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 8, 0, 0, 0, 0, 1, 0, 0, 0, 0, 8],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 8, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 8, 0, 0, 0],
    [0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 1, 8, 0, 8, 0, 1, 0],
    [0, 0, 0, 8, 8, 0, 0, 8, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 0, 0, 0, 0, 0, 0, 8],
], dtype=int)

T_OUT = np.array([
    [8, 8, 8, 8, 8, 8, 8, 8],
    [8, 0, 8, 0, 8, 0, 0, 8],
    [8, 0, 0, 0, 0, 0, 0, 8],
    [8, 0, 0, 0, 0, 0, 0, 8],
    [8, 0, 0, 0, 0, 0, 0, 8],
    [8, 0, 8, 0, 0, 0, 0, 8],
    [8, 0, 0, 0, 0, 8, 0, 8],
    [8, 8, 8, 8, 8, 8, 8, 8],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
	A=range;c,E=len(j),len(j[0]);k={}
	for W in A(c):
		for l in A(E):
			if j[W][l]:k[j[W][l]]=k.get(j[W][l],0)+1
	J,a=max(k,key=k.get),min(k,key=k.get);C,e=0,None
	for K in A(c-2):
		for w in A(E-2):
			for L in A(K+2,c):
				for b in A(w+2,E):
					if all(j[K][A]==J for A in A(w,b+1))and all(j[L][A]==J for A in A(w,b+1))and all(j[A][w]==J for A in A(K,L+1))and all(j[A][b]==J for A in A(K,L+1)):
						d=(L-K+1)*(b-w+1)
						if d>C:C,e=d,(K,w,L,b)
	K,w,L,b=e;return[[a if j[K+L][w+A]==J else j[K+L][w+A]for A in A(b-w+1)]for L in A(L-K+1)]


# --- Code Golf Solution (Compressed) ---
def q(m, n=266, *f):
    return [[sum({*e * sum(m, [-f])}) for e in s] for r in m if (f := (((X := (n >> 5)) * [(a := (r * 3)[(x := (n % 32))])] in ((s := r[x:x + X]), [f] * X)) * a))] or p(m, n - 1)


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

def interval(
    start: Integer,
    stop: Integer,
    step: Integer
) -> Tuple:
    """ range """
    return tuple(range(start, stop, step))

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

def subgrid(
    patch: Patch,
    grid: Grid
) -> Grid:
    """ smallest subgrid containing object """
    return crop(grid, ulcorner(patch), shape(patch))

def replace(
    grid: Grid,
    replacee: Integer,
    replacer: Integer
) -> Grid:
    """ color substitution """
    return tuple(tuple(replacer if v == replacee else v for v in r) for r in grid)

def canvas(
    value: Integer,
    dimensions: IntegerTuple
) -> Grid:
    """ grid construction """
    return tuple(tuple(value for j in range(dimensions[1])) for i in range(dimensions[0]))

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

def inbox(
    patch: Patch
) -> Indices:
    """ inbox for patch """
    ai, aj = uppermost(patch) + 1, leftmost(patch) + 1
    bi, bj = lowermost(patch) - 1, rightmost(patch) - 1
    si, sj = min(ai, bi), min(aj, bj)
    ei, ej = max(ai, bi), max(aj, bj)
    vlines = {(i, sj) for i in range(si, ei + 1)} | {(i, ej) for i in range(si, ei + 1)}
    hlines = {(si, j) for j in range(sj, ej + 1)} | {(ei, j) for j in range(sj, ej + 1)}
    return frozenset(vlines | hlines)

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

def generate_fcb5c309(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc, dotc, sqc = sample(cols, 3)
    numsq = unifint(diff_lb, diff_ub, (1, (h * w) // 25))
    gi = canvas(bgc, (h, w))
    inds = asindices(gi)
    maxtr = 4 * numsq
    tr = 0
    succ = 0
    numcells = None
    take = False
    while tr < maxtr and succ < numsq:
        oh = randint(3, 7)
        ow = randint(3, 7)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            break
        loc = choice(totuple(cands))
        loci, locj = loc
        sq = box(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
        bd = backdrop(sq)
        if bd.issubset(inds):
            gi = fill(gi, sqc, sq)
            ib = backdrop(inbox(sq))
            if numcells is None:
                numcells = unifint(diff_lb, diff_ub, (1, len(ib)))
                cells = sample(totuple(ib), numcells)
                take = True
            else:
                nc = unifint(diff_lb, diff_ub, (0, min(max(0, numcells - 1), len(ib))))
                cells = sample(totuple(ib), nc)
            gi = fill(gi, dotc, cells)
            if take:
                go = replace(subgrid(sq, gi), sqc, dotc)
                take = False
            inds = (inds - bd) - outbox(bd)
            succ += 1
        tr += 1
    nnoise = unifint(diff_lb, diff_ub, (0, max(0, len(inds) // 2 - 1)))
    noise = sample(totuple(inds), nnoise)
    gi = fill(gi, dotc, noise)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

IntegerSet = FrozenSet[Integer]

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

TWO = 2

F = False

T = True

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

def contained(
    value: Any,
    container: Container
) -> Boolean:
    """ element of """
    return value in container

def greater(
    a: Integer,
    b: Integer
) -> Boolean:
    """ greater """
    return a > b

def minimum(
    container: IntegerSet
) -> Integer:
    """ minimum """
    return min(container, default=0)

def argmax(
    container: Container,
    compfunc: Callable
) -> Any:
    """ largest item by custom order """
    return max(container, key=compfunc, default=None)

def argmin(
    container: Container,
    compfunc: Callable
) -> Any:
    """ smallest item by custom order """
    return min(container, key=compfunc, default=None)

def both(
    a: Boolean,
    b: Boolean
) -> Boolean:
    """ logical and """
    return a and b

def extract(
    container: Container,
    condition: Callable
) -> Any:
    """ first element of container that satisfies condition """
    return next(e for e in container if condition(e))

def remove(
    value: Any,
    container: Container
) -> Container:
    """ remove item from container """
    return type(container)(e for e in container if e != value)

def compose(
    outer: Callable,
    inner: Callable
) -> Callable:
    """ function composition """
    return lambda x: outer(inner(x))

def chain(
    h: Callable,
    g: Callable,
    f: Callable
) -> Callable:
    """ function composition with three functions """
    return lambda x: h(g(f(x)))

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

def mostcolor(
    element: Element
) -> Integer:
    """ most common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return max(set(values), key=values.count)

def colorcount(
    element: Element,
    value: Integer
) -> Integer:
    """ number of cells with color """
    if isinstance(element, tuple):
        return sum(row.count(value) for row in element)
    return sum(v == value for v, _ in element)

def colorfilter(
    objs: Objects,
    value: Integer
) -> Objects:
    """ filter objects by color """
    return frozenset(obj for obj in objs if next(iter(obj))[0] == value)

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

def palette(
    element: Element
) -> IntegerSet:
    """ colors occurring in object or grid """
    if isinstance(element, tuple):
        return frozenset({v for r in element for v in r})
    return frozenset({v for v, _ in element})

def toobject(
    patch: Patch,
    grid: Grid
) -> Object:
    """ object from patch and grid """
    h, w = len(grid), len(grid[0])
    return frozenset((grid[i][j], (i, j)) for i, j in toindices(patch) if 0 <= i < h and 0 <= j < w)

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_fcb5c309(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = objects(I, T, F, F)
    x1 = lbind(contained, F)
    x2 = compose(flip, x1)
    x3 = fork(equality, toindices, box)
    x4 = lbind(apply, x3)
    x5 = lbind(colorfilter, x0)
    x6 = chain(x2, x4, x5)
    x7 = rbind(greater, TWO)
    x8 = compose(minimum, shape)
    x9 = lbind(apply, x8)
    x10 = chain(x7, minimum, x9)
    x11 = lbind(colorfilter, x0)
    x12 = compose(x10, x11)
    x13 = fork(both, x6, x12)
    x14 = palette(I)
    x15 = extract(x14, x13)
    x16 = palette(I)
    x17 = remove(x15, x16)
    x18 = lbind(colorcount, I)
    x19 = argmin(x17, x18)
    x20 = rbind(colorcount, x19)
    x21 = rbind(toobject, I)
    x22 = chain(x20, x21, backdrop)
    x23 = colorfilter(x0, x15)
    x24 = argmax(x23, x22)
    x25 = subgrid(x24, I)
    x26 = replace(x25, x15, x19)
    return x26


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_fcb5c309(inp)
        assert pred == _to_grid(expected), f"{name} failed"

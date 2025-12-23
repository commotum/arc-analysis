# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "e50d258f"
SERIAL = "365"
URL    = "https://arcprize.org/play?task=e50d258f"

# --- Code Golf Concepts ---
CONCEPTS = [
    "separate_images",
    "detect_background_color",
    "crop",
    "count_tiles",
    "take_maximum",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 0, 0, 0, 0, 8, 8, 8, 8],
    [0, 8, 8, 8, 8, 0, 8, 2, 2, 8],
    [0, 8, 1, 8, 8, 0, 8, 8, 8, 8],
    [0, 8, 8, 2, 8, 0, 8, 2, 1, 8],
    [0, 8, 8, 8, 8, 0, 8, 8, 8, 8],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 8, 8, 8, 8, 8, 0],
    [0, 0, 0, 8, 8, 8, 2, 8, 8, 0],
    [0, 0, 0, 8, 2, 8, 1, 8, 8, 0],
    [0, 0, 0, 8, 1, 8, 8, 8, 8, 0],
], dtype=int)

E1_OUT = np.array([
    [8, 8, 8, 8],
    [8, 2, 2, 8],
    [8, 8, 8, 8],
    [8, 2, 1, 8],
    [8, 8, 8, 8],
], dtype=int)

E2_IN = np.array([
    [1, 1, 1, 8, 0, 0, 0, 0, 0, 0],
    [1, 8, 1, 1, 0, 1, 8, 8, 1, 8],
    [8, 2, 8, 1, 0, 8, 1, 8, 2, 8],
    [1, 1, 1, 8, 0, 8, 8, 8, 8, 1],
    [8, 1, 8, 8, 0, 8, 1, 2, 8, 2],
    [0, 0, 0, 0, 0, 8, 8, 8, 1, 8],
    [0, 0, 0, 0, 0, 1, 1, 8, 1, 8],
    [0, 8, 2, 2, 0, 8, 1, 1, 8, 2],
    [0, 2, 2, 1, 0, 0, 0, 0, 0, 0],
    [0, 2, 1, 8, 0, 0, 0, 0, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [8, 2, 2],
    [2, 2, 1],
    [2, 1, 8],
], dtype=int)

E3_IN = np.array([
    [2, 8, 8, 8, 0, 0, 0, 0, 0, 0],
    [8, 8, 1, 8, 0, 0, 0, 0, 0, 0],
    [1, 8, 8, 8, 0, 0, 0, 0, 0, 0],
    [8, 8, 8, 2, 0, 0, 1, 8, 8, 2],
    [8, 2, 8, 1, 0, 0, 8, 8, 1, 8],
    [8, 1, 8, 8, 0, 0, 8, 2, 8, 8],
    [0, 0, 0, 0, 0, 0, 8, 8, 8, 1],
    [0, 0, 0, 0, 0, 0, 1, 8, 8, 8],
    [0, 0, 0, 0, 0, 0, 8, 8, 1, 8],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [2, 8, 8, 8],
    [8, 8, 1, 8],
    [1, 8, 8, 8],
    [8, 8, 8, 2],
    [8, 2, 8, 1],
    [8, 1, 8, 8],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [2, 8, 8, 8, 0, 0, 0, 0, 0, 0],
    [8, 8, 1, 8, 0, 0, 2, 8, 1, 0],
    [1, 2, 8, 1, 0, 0, 8, 8, 8, 0],
    [8, 8, 8, 8, 0, 0, 2, 1, 8, 0],
    [0, 0, 0, 0, 0, 0, 8, 8, 2, 0],
    [0, 0, 0, 0, 0, 0, 2, 8, 1, 0],
    [0, 1, 2, 8, 2, 0, 1, 8, 8, 0],
    [0, 8, 8, 1, 8, 0, 0, 0, 0, 0],
    [0, 1, 2, 8, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

T_OUT = np.array([
    [2, 8, 1],
    [8, 8, 8],
    [2, 1, 8],
    [8, 8, 2],
    [2, 8, 1],
    [1, 8, 8],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
	A,c=len(j),len(j[0]);E=-1
	for k in range(A):
		for W in range(c):
			if j[k][W]and(k<1 or j[k-1][W]<1)and(W<1 or j[k][W-1]<1):
				l=J=1
				while W+l<c and j[k][W+l]:l+=1
				while k+J<A and j[k+J][W]:J+=1
				a=[k[W:W+l]for k in j[k:k+J]];C=sum(k.count(2)for k in a)
				if C>E:E=C;e=a
	return e


# --- Code Golf Solution (Compressed) ---
def q(g):
    return max(((-(y := sum((s := [r[x % 9:x % 13] for r in g[x % 8:x % 11]]), g).count)(0), y(2), y(1), s) for x in range(5 ** 6)))[3]


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

def generate_e50d258f(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(2, interval(0, 10, 1))    
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    padcol = choice(remcols)
    remcols = remove(padcol, remcols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    num = unifint(diff_lb, diff_ub, (1, 10))
    indss = asindices(gi)
    maxtrials = 4 * num
    tr = 0
    succ = 0
    bound = None
    go = None
    while succ < num and tr <= maxtrials:
        if len(remcols) == 0 or len(indss) == 0:
            break
        oh = randint(3, 8)
        ow = randint(3, 8)
        subs = totuple(sfilter(indss, lambda ij: ij[0] < h - oh and ij[1] < w - ow))
        if len(subs) == 0:
            tr += 1
            continue
        loci, locj = choice(subs)
        obj = frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)})
        bd = backdrop(obj)
        if bd.issubset(indss):
            numcc = unifint(diff_lb, diff_ub, (1, 7))
            ccols = sample(remcols, numcc)
            if succ == 0:
                numred = unifint(diff_lb, diff_ub, (1, oh * ow))
                bound = numred
            else:
                numred = unifint(diff_lb, diff_ub, (0, min(oh * ow, bound - 1)))
            cc = canvas(choice(ccols), (oh, ow))
            cci = asindices(cc)
            subs = sample(totuple(cci), numred)
            obj1 = {(choice(ccols), ij) for ij in cci - set(subs)}
            obj2 = {(2, ij) for ij in subs}
            obj = obj1 | obj2
            gi = paint(gi, shift(obj, (loci, locj)))
            if go is None:
                go = paint(cc, obj)
            succ += 1
            indss = (indss - bd) - outbox(bd)
        tr += 1
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

TWO = 2

F = False

T = True

def argmax(
    container: Container,
    compfunc: Callable
) -> Any:
    """ largest item by custom order """
    return max(container, key=compfunc, default=None)

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

def colorcount(
    element: Element,
    value: Integer
) -> Integer:
    """ number of cells with color """
    if isinstance(element, tuple):
        return sum(row.count(value) for row in element)
    return sum(v == value for v, _ in element)

def crop(
    grid: Grid,
    start: IntegerTuple,
    dims: IntegerTuple
) -> Grid:
    """ subgrid specified by start and dimension """
    return tuple(r[start[1]:start[1]+dims[1]] for r in grid[start[0]:start[0]+dims[0]])

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

def toobject(
    patch: Patch,
    grid: Grid
) -> Object:
    """ object from patch and grid """
    h, w = len(grid), len(grid[0])
    return frozenset((grid[i][j], (i, j)) for i, j in toindices(patch) if 0 <= i < h and 0 <= j < w)

def hconcat(
    a: Grid,
    b: Grid
) -> Grid:
    """ concatenate two grids horizontally """
    return tuple(i + j for i, j in zip(a, b))

def subgrid(
    patch: Patch,
    grid: Grid
) -> Grid:
    """ smallest subgrid containing object """
    return crop(grid, ulcorner(patch), shape(patch))

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

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_e50d258f(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = asindices(I)
    x1 = box(x0)
    x2 = toobject(x1, I)
    x3 = mostcolor(x2)
    x4 = shape(I)
    x5 = canvas(x3, x4)
    x6 = hconcat(I, x5)
    x7 = objects(x6, F, F, T)
    x8 = rbind(colorcount, TWO)
    x9 = argmax(x7, x8)
    x10 = subgrid(x9, I)
    return x10


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_e50d258f(inp)
        assert pred == _to_grid(expected), f"{name} failed"

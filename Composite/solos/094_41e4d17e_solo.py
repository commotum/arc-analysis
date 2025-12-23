# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "41e4d17e"
SERIAL = "094"
URL    = "https://arcprize.org/play?task=41e4d17e"

# --- Code Golf Concepts ---
CONCEPTS = [
    "draw_line_from_point",
    "pattern_repetition",
]

# --- Example Grids ---
E1_IN = np.array([
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 1, 8, 8, 8, 1, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 1, 8, 8, 8, 1, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 1, 8, 8, 8, 1, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
], dtype=int)

E1_OUT = np.array([
    [8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 1, 8, 6, 8, 1, 8, 8, 8, 8, 8, 8, 8],
    [6, 6, 6, 1, 6, 6, 6, 1, 6, 6, 6, 6, 6, 6, 6],
    [8, 8, 8, 1, 8, 6, 8, 1, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8],
], dtype=int)

E2_IN = np.array([
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 1, 8, 8, 8, 1, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 1, 8, 8, 8, 1, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 1, 8, 8, 8, 1, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 1, 1, 1, 1, 1, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 1, 8, 8, 8, 1, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 1, 8, 8, 8, 1, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 1, 8, 8, 8, 1, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 1, 1, 1, 1, 1, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
], dtype=int)

E2_OUT = np.array([
    [8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 6, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 6, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 6, 8, 8, 8, 8],
    [8, 8, 8, 1, 1, 1, 1, 1, 8, 8, 6, 8, 8, 8, 8],
    [8, 8, 8, 1, 8, 6, 8, 1, 8, 8, 6, 8, 8, 8, 8],
    [6, 6, 6, 1, 6, 6, 6, 1, 6, 6, 6, 6, 6, 6, 6],
    [8, 8, 8, 1, 8, 6, 8, 1, 8, 8, 6, 8, 8, 8, 8],
    [8, 8, 8, 1, 1, 1, 1, 1, 8, 8, 6, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 6, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 6, 8, 8, 1, 1, 1, 1, 1, 8, 8],
    [8, 8, 8, 8, 8, 6, 8, 8, 1, 8, 6, 8, 1, 8, 8],
    [6, 6, 6, 6, 6, 6, 6, 6, 1, 6, 6, 6, 1, 6, 6],
    [8, 8, 8, 8, 8, 6, 8, 8, 1, 8, 6, 8, 1, 8, 8],
    [8, 8, 8, 8, 8, 6, 8, 8, 1, 1, 1, 1, 1, 8, 8],
    [8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 6, 8, 8, 8, 8],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 1, 1, 1, 1, 1, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 1, 8, 8, 8, 1, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 1, 8, 8, 8, 1, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 1, 8, 8, 8, 1, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 1, 1, 1, 1, 1, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 1, 8, 8, 8, 1, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 1, 8, 8, 8, 1, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 1, 8, 8, 8, 1, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
], dtype=int)

T_OUT = np.array([
    [8, 8, 8, 8, 8, 6, 8, 8, 6, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 6, 1, 1, 1, 1, 1, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 6, 1, 8, 6, 8, 1, 8, 8, 8, 8],
    [6, 6, 6, 6, 6, 6, 1, 6, 6, 6, 1, 6, 6, 6, 6],
    [8, 8, 8, 8, 8, 6, 1, 8, 6, 8, 1, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 6, 1, 1, 1, 1, 1, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 6, 8, 8, 6, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 6, 8, 8, 6, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 6, 8, 8, 6, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 1, 1, 1, 1, 1, 6, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 1, 8, 6, 8, 1, 6, 8, 8, 8, 8, 8, 8],
    [6, 6, 6, 1, 6, 6, 6, 1, 6, 6, 6, 6, 6, 6, 6],
    [8, 8, 8, 1, 8, 6, 8, 1, 6, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 1, 1, 1, 1, 1, 6, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 6, 8, 8, 6, 8, 8, 8, 8, 8, 8],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
j=len
A=range
def p(c):
	E,k=[],[];W,l=j(c),j(c[0])
	for J in A(W-4):
		for a in A(l-4):
			C=[[c[E+J][C+a]for E in A(5)]for C in A(5)];C=[a for J in C for a in J];C=sum([J for J in C if J==1])
			if C==16:E.append(J+2);k.append(a+2)
	for J in A(W):
		for a in A(l):
			if J in E or a in k:
				if c[J][a]!=1:c[J][a]=6
	return c


# --- Code Golf Solution (Compressed) ---
def q(g, x=0):
    return eval(re.sub('8(?=[^(]*+[^)]*1.{46}1, 1)', '6', f'{(*zip(*(x or p(g, g))),)}'))


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

def asindices(
    grid: Grid
) -> Indices:
    """ indices of all grid cells """
    return frozenset((i, j) for i in range(len(grid)) for j in range(len(grid[0])))

def ofcolor(
    grid: Grid,
    value: Integer
) -> Indices:
    """ indices of all grid cells with value """
    return frozenset((i, j) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value)

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

def center(
    patch: Patch
) -> IntegerTuple:
    """ center of the patch """
    return (uppermost(patch) + height(patch) // 2, leftmost(patch) + width(patch) // 2)

def canvas(
    value: Integer,
    dimensions: IntegerTuple
) -> Grid:
    """ grid construction """
    return tuple(tuple(value for j in range(dimensions[1])) for i in range(dimensions[0]))

def vfrontier(
    location: IntegerTuple
) -> Indices:
    """ vertical frontier """
    return frozenset((i, location[1]) for i in range(30))

def hfrontier(
    location: IntegerTuple
) -> Indices:
    """ horizontal frontier """
    return frozenset((location[0], j) for j in range(30))

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

def generate_41e4d17e(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(6, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    num = unifint(diff_lb, diff_ub, (1, (h * w) // 16))
    bgc, fgc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    bx = box(frozenset({(0, 0), (4, 4)}))
    bd = backdrop(bx)
    maxtrials = 4 * num
    succ = 0
    tr = 0
    while succ < num and tr < maxtrials:
        loc = choice(totuple(inds))
        bxs = shift(bx, loc)
        if bxs.issubset(set(inds)):
            gi = fill(gi, fgc, bxs)
            go = fill(go, fgc, bxs)
            cen = center(bxs)
            frns = hfrontier(cen) | vfrontier(cen)
            kep = frns & ofcolor(go, bgc)
            go = fill(go, 6, kep)
            inds = difference(inds, shift(bd, loc))
            succ += 1
        tr += 1
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

ContainerContainer = Container[Container]

SIX = 6

NINE = 9

F = False

T = True

def equality(
    a: Any,
    b: Any
) -> Boolean:
    """ equality """
    return a == b

def combine(
    a: Container,
    b: Container
) -> Container:
    """ union """
    return type(a)((*a, *b))

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

def both(
    a: Boolean,
    b: Boolean
) -> Boolean:
    """ logical and """
    return a and b

def sfilter(
    container: Container,
    condition: Callable
) -> Container:
    """ keep elements in container that satisfy condition """
    return type(container)(e for e in container if condition(e))

def compose(
    outer: Callable,
    inner: Callable
) -> Callable:
    """ function composition """
    return lambda x: outer(inner(x))

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

def underfill(
    grid: Grid,
    value: Integer,
    patch: Patch
) -> Grid:
    """ fill value at indices that are background """
    h, w = len(grid), len(grid[0])
    bg = mostcolor(grid)
    grid_filled = list(list(row) for row in grid)
    for i, j in toindices(patch):
        if 0 <= i < h and 0 <= j < w:
            if grid_filled[i][j] == bg:
                grid_filled[i][j] = value
    return tuple(tuple(row) for row in grid_filled)

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_41e4d17e(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = lbind(equality, NINE)
    x1 = compose(x0, size)
    x2 = fork(equality, height, width)
    x3 = fork(both, x1, x2)
    x4 = objects(I, T, F, F)
    x5 = sfilter(x4, x3)
    x6 = fork(combine, vfrontier, hfrontier)
    x7 = compose(x6, center)
    x8 = mapply(x7, x5)
    x9 = underfill(I, SIX, x8)
    return x9


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_41e4d17e(inp)
        assert pred == _to_grid(expected), f"{name} failed"

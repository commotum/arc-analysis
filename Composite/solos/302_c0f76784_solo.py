# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "c0f76784"
SERIAL = "302"
URL    = "https://arcprize.org/play?task=c0f76784"

# --- Code Golf Concepts ---
CONCEPTS = [
    "loop_filling",
    "measure_area",
    "associate_colors_to_numbers",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5],
    [0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 5],
    [0, 0, 5, 5, 5, 0, 0, 5, 0, 0, 0, 5],
    [0, 0, 5, 0, 5, 0, 0, 5, 0, 0, 0, 5],
    [0, 0, 5, 5, 5, 0, 0, 5, 5, 5, 5, 5],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 0, 0],
    [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0, 0],
    [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0, 0],
    [0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5],
    [0, 0, 0, 0, 0, 0, 0, 5, 8, 8, 8, 5],
    [0, 0, 5, 5, 5, 0, 0, 5, 8, 8, 8, 5],
    [0, 0, 5, 6, 5, 0, 0, 5, 8, 8, 8, 5],
    [0, 0, 5, 5, 5, 0, 0, 5, 5, 5, 5, 5],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 0, 0],
    [0, 0, 0, 0, 0, 0, 5, 7, 7, 5, 0, 0],
    [0, 0, 0, 0, 0, 0, 5, 7, 7, 5, 0, 0],
    [0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E2_IN = np.array([
    [0, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0],
    [0, 5, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0],
    [0, 5, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0],
    [0, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 0],
    [0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 5, 0],
    [0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 5, 0],
    [0, 5, 5, 5, 0, 0, 5, 0, 0, 0, 5, 0],
    [0, 5, 0, 5, 0, 0, 5, 5, 5, 5, 5, 0],
    [0, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [0, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0],
    [0, 5, 7, 7, 5, 0, 0, 0, 0, 0, 0, 0],
    [0, 5, 7, 7, 5, 0, 0, 0, 0, 0, 0, 0],
    [0, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 0],
    [0, 0, 0, 0, 0, 0, 5, 8, 8, 8, 5, 0],
    [0, 0, 0, 0, 0, 0, 5, 8, 8, 8, 5, 0],
    [0, 5, 5, 5, 0, 0, 5, 8, 8, 8, 5, 0],
    [0, 5, 6, 5, 0, 0, 5, 5, 5, 5, 5, 0],
    [0, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E3_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0],
    [0, 5, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
    [0, 5, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
    [0, 5, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
    [0, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 5, 5, 5, 5, 0, 0, 0, 0],
    [0, 0, 0, 0, 5, 0, 0, 5, 0, 0, 0, 0],
    [0, 0, 0, 0, 5, 0, 0, 5, 0, 0, 0, 0],
    [0, 0, 0, 0, 5, 5, 5, 5, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0],
    [0, 5, 8, 8, 8, 5, 0, 0, 0, 0, 0, 0],
    [0, 5, 8, 8, 8, 5, 0, 0, 0, 0, 0, 0],
    [0, 5, 8, 8, 8, 5, 0, 0, 0, 0, 0, 0],
    [0, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 5, 5, 5, 5, 0, 0, 0, 0],
    [0, 0, 0, 0, 5, 7, 7, 5, 0, 0, 0, 0],
    [0, 0, 0, 0, 5, 7, 7, 5, 0, 0, 0, 0],
    [0, 0, 0, 0, 5, 5, 5, 5, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0],
    [0, 5, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
    [0, 5, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
    [0, 5, 0, 0, 0, 5, 0, 0, 5, 5, 5, 0],
    [0, 5, 5, 5, 5, 5, 0, 0, 5, 0, 5, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 5, 5, 5, 5, 0, 0, 0, 0],
    [0, 0, 0, 0, 5, 0, 0, 5, 0, 0, 0, 0],
    [0, 0, 0, 0, 5, 0, 0, 5, 0, 0, 0, 0],
    [0, 0, 0, 0, 5, 5, 5, 5, 0, 0, 0, 0],
], dtype=int)

T_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0],
    [0, 5, 8, 8, 8, 5, 0, 0, 0, 0, 0, 0],
    [0, 5, 8, 8, 8, 5, 0, 0, 0, 0, 0, 0],
    [0, 5, 8, 8, 8, 5, 0, 0, 5, 5, 5, 0],
    [0, 5, 5, 5, 5, 5, 0, 0, 5, 6, 5, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 5, 5, 5, 5, 0, 0, 0, 0],
    [0, 0, 0, 0, 5, 7, 7, 5, 0, 0, 0, 0],
    [0, 0, 0, 0, 5, 7, 7, 5, 0, 0, 0, 0],
    [0, 0, 0, 0, 5, 5, 5, 5, 0, 0, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
	A,c=len(j),len(j[0]);E=[[0]*c for b in j];k=[]
	def e(W,l):
		J=[(W,l)];E[W][l]=1;a=[(W,l)];C=1
		while J:
			e,K=J.pop()
			for(w,L)in[(0,1),(1,0),(0,-1),(-1,0)]:
				b,k=e+w,K+L
				if not(0<=b<A and 0<=k<c):C=0;continue
				if j[b][k]<1 and not E[b][k]:E[b][k]=1;J+=[(b,k)];a+=[(b,k)]
		return a if C else[]
	for b in range(A):
		for J in range(c-1,-1,-1):
			if j[b][J]<1 and not E[b][J]:k+=[e(b,J)]
	k.sort(key=len,reverse=1)
	for(b,a)in enumerate(k):
		K=min(8,max(6,len(a)**.5+.0+5))
		for C in a:j[C[0]][C[1]]=K
	return j


# --- Code Golf Solution (Compressed) ---
def q(g):
    return eval(re.sub('([^5]..5,)([^5]+)', '\\1*(k:=len([\\2]))*[k+5],', str(g)))


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

def generate_c0f76784(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (6, 7, 8))    
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, len(remcols)))
    ccols = sample(remcols, numcols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    num = unifint(diff_lb, diff_ub, (1, (h * w) // 20))
    indss = asindices(gi)
    maxtrials = 4 * num
    tr = 0
    succ = 0
    while succ < num and tr <= maxtrials:
        if len(indss) == 0:
            break
        oh = choice((3, 4, 5))
        ow = oh
        subs = totuple(sfilter(indss, lambda ij: ij[0] < h - oh and ij[1] < w - ow))
        if len(subs) == 0:
            tr += 1
            continue
        loci, locj = choice(subs)
        obj = frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)})
        bd = backdrop(obj)
        col = choice(ccols)
        if bd.issubset(indss):
            gi = fill(gi, col, bd)
            go = fill(go, col, bd)
            ccc = oh + 3
            bdx = backdrop(inbox(obj))
            gi = fill(gi, bgc, bdx)
            go = fill(go, ccc, bdx)
            succ += 1
            indss = (indss - bd) - outbox(bd)
        tr += 1
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

ContainerContainer = Container[Container]

ONE = 1

FOUR = 4

SIX = 6

SEVEN = 7

EIGHT = 8

NINE = 9

F = False

T = True

def merge(
    containers: ContainerContainer
) -> Container:
    """ merging """
    return type(containers)(e for c in containers for e in c)

def mostcolor(
    element: Element
) -> Integer:
    """ most common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return max(set(values), key=values.count)

def colorfilter(
    objs: Objects,
    value: Integer
) -> Objects:
    """ filter objects by color """
    return frozenset(obj for obj in objs if next(iter(obj))[0] == value)

def sizefilter(
    container: Container,
    n: Integer
) -> FrozenSet:
    """ filter items by size """
    return frozenset(item for item in container if len(item) == n)

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

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_c0f76784(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = objects(I, T, F, F)
    x1 = mostcolor(I)
    x2 = colorfilter(x0, x1)
    x3 = sizefilter(x2, ONE)
    x4 = merge(x3)
    x5 = sizefilter(x2, FOUR)
    x6 = merge(x5)
    x7 = sizefilter(x2, NINE)
    x8 = merge(x7)
    x9 = fill(I, SIX, x4)
    x10 = fill(x9, SEVEN, x6)
    x11 = fill(x10, EIGHT, x8)
    return x11


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_c0f76784(inp)
        assert pred == _to_grid(expected), f"{name} failed"

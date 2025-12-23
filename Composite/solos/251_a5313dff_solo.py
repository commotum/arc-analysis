# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "a5313dff"
SERIAL = "251"
URL    = "https://arcprize.org/play?task=a5313dff"

# --- Code Golf Concepts ---
CONCEPTS = [
    "loop_filling",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 2, 2, 2, 2, 0, 0],
    [0, 2, 0, 0, 0, 2, 0, 0],
    [0, 2, 0, 2, 0, 2, 0, 0],
    [0, 2, 0, 0, 0, 2, 0, 0],
    [0, 2, 2, 2, 2, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 2, 2, 2, 2, 0, 0],
    [0, 2, 1, 1, 1, 2, 0, 0],
    [0, 2, 1, 2, 1, 2, 0, 0],
    [0, 2, 1, 1, 1, 2, 0, 0],
    [0, 2, 2, 2, 2, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E2_IN = np.array([
    [0, 0, 0, 2, 0, 0, 0, 0],
    [0, 2, 0, 2, 0, 0, 0, 0],
    [0, 0, 0, 2, 0, 0, 0, 0],
    [2, 2, 2, 2, 2, 2, 2, 0],
    [0, 0, 2, 0, 0, 0, 2, 0],
    [0, 0, 2, 0, 2, 0, 2, 0],
    [0, 0, 2, 0, 0, 0, 2, 0],
    [0, 0, 2, 2, 2, 2, 2, 0],
], dtype=int)

E2_OUT = np.array([
    [0, 0, 0, 2, 0, 0, 0, 0],
    [0, 2, 0, 2, 0, 0, 0, 0],
    [0, 0, 0, 2, 0, 0, 0, 0],
    [2, 2, 2, 2, 2, 2, 2, 0],
    [0, 0, 2, 1, 1, 1, 2, 0],
    [0, 0, 2, 1, 2, 1, 2, 0],
    [0, 0, 2, 1, 1, 1, 2, 0],
    [0, 0, 2, 2, 2, 2, 2, 0],
], dtype=int)

E3_IN = np.array([
    [0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 0, 0],
    [0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0],
    [0, 0, 0, 2, 0, 2, 0, 2, 2, 2, 2, 0],
    [0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 2, 0],
    [0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 2, 0],
    [0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 2, 0],
    [0, 0, 0, 2, 0, 2, 0, 2, 2, 2, 2, 0],
    [0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0],
    [0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2],
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 2],
], dtype=int)

E3_OUT = np.array([
    [0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 0, 0],
    [0, 0, 0, 2, 1, 1, 1, 2, 0, 0, 0, 0],
    [0, 0, 0, 2, 1, 2, 1, 2, 2, 2, 2, 0],
    [0, 0, 0, 2, 1, 1, 1, 2, 1, 1, 2, 0],
    [0, 0, 0, 2, 2, 2, 2, 2, 1, 1, 2, 0],
    [0, 0, 0, 2, 1, 1, 1, 2, 1, 1, 2, 0],
    [0, 0, 0, 2, 1, 2, 1, 2, 2, 2, 2, 0],
    [0, 0, 0, 2, 1, 1, 1, 2, 0, 0, 0, 0],
    [0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2],
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 2],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 2, 2, 2, 2, 2, 0, 0],
    [0, 0, 2, 0, 0, 0, 2, 0, 0],
    [0, 0, 2, 0, 0, 0, 2, 0, 0],
    [2, 2, 2, 2, 2, 2, 2, 0, 0],
    [2, 0, 0, 0, 2, 0, 0, 0, 0],
    [2, 0, 2, 0, 2, 0, 0, 0, 0],
    [2, 0, 0, 0, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 0, 0, 0, 0],
], dtype=int)

T_OUT = np.array([
    [0, 0, 2, 2, 2, 2, 2, 0, 0],
    [0, 0, 2, 1, 1, 1, 2, 0, 0],
    [0, 0, 2, 1, 1, 1, 2, 0, 0],
    [2, 2, 2, 2, 2, 2, 2, 0, 0],
    [2, 1, 1, 1, 2, 0, 0, 0, 0],
    [2, 1, 2, 1, 2, 0, 0, 0, 0],
    [2, 1, 1, 1, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 0, 0, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j,A=range):
	c,E=len(j),len(j[0]);k=[[0]*E for c in A(c)];W=[]
	for l in A(c):
		for J in A(E):
			if l*J==0 or l==c-1 or J==E-1:
				if j[l][J]==0:k[l][J]=1;W.append((l,J))
	while W:
		a,C=W.pop(0)
		for(e,K)in[(-1,0),(1,0),(0,-1),(0,1)]:
			w,L=a+e,C+K
			if 0<=w<c and 0<=L<E and j[w][L]==0 and not k[w][L]:k[w][L]=1;W.append((w,L))
	b=[[j[c][E]if j[c][E]!=0 or k[c][E]else 1 for E in A(E)]for c in A(c)];return b


# --- Code Golf Solution (Compressed) ---
def q(g, i=31):
    return g * -i or p([[r.pop() & ~4 ** [0, *r][-1] or i > 30 for r in g] for _ in g], i - 1)


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, randint, sample, uniform

Boolean = bool

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Objects = FrozenSet[Object]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

Element = Union[Object, Grid]

ContainerContainer = Container[Container]

F = False

T = True

def flip(
    b: Boolean
) -> Boolean:
    """ logical not """
    return not b

def merge(
    containers: ContainerContainer
) -> Container:
    """ merging """
    return type(containers)(e for c in containers for e in c)

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

def compose(
    outer: Callable,
    inner: Callable
) -> Callable:
    """ function composition """
    return lambda x: outer(inner(x))

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

def colorfilter(
    objs: Objects,
    value: Integer
) -> Objects:
    """ filter objects by color """
    return frozenset(obj for obj in objs if next(iter(obj))[0] == value)

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

def bordering(
    patch: Patch,
    grid: Grid
) -> Boolean:
    """ whether a patch is adjacent to a grid border """
    return uppermost(patch) == 0 or leftmost(patch) == 0 or lowermost(patch) == len(grid) - 1 or rightmost(patch) == len(grid[0]) - 1

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

def generate_a5313dff(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(1, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc, fgc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    noccs = unifint(diff_lb, diff_ub, (1, (h * w) // 20))
    succ = 0
    tr = 0
    maxtr = 10 * noccs
    inds = shift(asindices(canvas(-1, (h+2, w+2))), (-1, -1))
    while (tr < maxtr and succ < noccs) or len(sfilter(colorfilter(objects(gi, T, F, F), bgc), compose(flip, rbind(bordering, gi)))) == 0:
        tr += 1
        oh = randint(3, 8)
        ow = randint(3, 8)
        bx = box(frozenset({(0, 0), (oh - 1, ow - 1)}))
        ins = backdrop(inbox(bx))
        loc = choice(totuple(inds))
        plcdins = shift(ins, loc)
        if len(plcdins & ofcolor(gi, fgc)) == 0:
            succ += 1
            gi = fill(gi, fgc, shift(bx, loc))
            if choice((True, True, False)):
                ss = sample(totuple(plcdins), randint(1, max(1, len(ins) // 2)))
                gi = fill(gi, fgc, ss)
    res = mfilter(colorfilter(objects(gi, T, F, F), bgc), compose(flip, rbind(bordering, gi)))
    go = fill(gi, 1, res)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
ONE = 1

def mostcommon(
    container: Container
) -> Any:
    """ most common item """
    return max(set(container), key=container.count)

def apply(
    function: Callable,
    container: Container
) -> Container:
    """ apply function to each item in container """
    return type(container)(function(e) for e in container)

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

def verify_a5313dff(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = objects(I, T, F, F)
    x1 = rbind(bordering, I)
    x2 = compose(flip, x1)
    x3 = sfilter(x0, x2)
    x4 = totuple(x3)
    x5 = apply(color, x4)
    x6 = mostcommon(x5)
    x7 = mostcolor(I)
    x8 = colorfilter(x0, x7)
    x9 = rbind(bordering, I)
    x10 = compose(flip, x9)
    x11 = mfilter(x8, x10)
    x12 = fill(I, ONE, x11)
    return x12


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_a5313dff(inp)
        assert pred == _to_grid(expected), f"{name} failed"

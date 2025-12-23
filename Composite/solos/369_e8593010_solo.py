# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "e8593010"
SERIAL = "369"
URL    = "https://arcprize.org/play?task=e8593010"

# --- Code Golf Concepts ---
CONCEPTS = [
    "holes",
    "count_tiles",
    "loop_filling",
    "associate_colors_to_numbers",
]

# --- Example Grids ---
E1_IN = np.array([
    [5, 5, 5, 5, 0, 5, 5, 5, 0, 5],
    [0, 0, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 5, 5, 5, 5, 5, 0, 0, 5, 0],
    [5, 5, 0, 5, 5, 5, 5, 0, 5, 0],
    [5, 5, 5, 5, 0, 0, 5, 5, 5, 5],
    [0, 5, 0, 5, 5, 5, 5, 0, 5, 0],
    [0, 5, 5, 5, 0, 0, 5, 5, 5, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 0],
    [0, 5, 5, 5, 5, 5, 5, 0, 5, 0],
], dtype=int)

E1_OUT = np.array([
    [5, 5, 5, 5, 3, 5, 5, 5, 3, 5],
    [1, 1, 5, 5, 5, 5, 5, 5, 5, 5],
    [1, 5, 5, 5, 5, 5, 1, 1, 5, 2],
    [5, 5, 3, 5, 5, 5, 5, 1, 5, 2],
    [5, 5, 5, 5, 2, 2, 5, 5, 5, 5],
    [2, 5, 3, 5, 5, 5, 5, 3, 5, 2],
    [2, 5, 5, 5, 2, 2, 5, 5, 5, 2],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 2],
    [3, 5, 5, 5, 5, 5, 5, 3, 5, 2],
], dtype=int)

E2_IN = np.array([
    [5, 5, 5, 5, 5, 0, 0, 5, 5, 5],
    [0, 0, 5, 0, 5, 5, 5, 5, 5, 0],
    [5, 5, 5, 5, 5, 0, 5, 0, 0, 5],
    [5, 0, 5, 5, 5, 0, 5, 5, 5, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 0, 5],
    [5, 5, 5, 5, 0, 5, 5, 5, 5, 5],
    [0, 0, 5, 5, 0, 5, 0, 0, 5, 0],
    [5, 5, 5, 5, 5, 5, 5, 0, 5, 5],
    [0, 5, 5, 5, 5, 5, 0, 5, 5, 0],
    [0, 0, 5, 5, 5, 5, 5, 5, 0, 5],
], dtype=int)

E2_OUT = np.array([
    [5, 5, 5, 5, 5, 2, 2, 5, 5, 5],
    [2, 2, 5, 3, 5, 5, 5, 5, 5, 3],
    [5, 5, 5, 5, 5, 2, 5, 2, 2, 5],
    [5, 3, 5, 5, 5, 2, 5, 5, 5, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 3, 5],
    [5, 5, 5, 5, 2, 5, 5, 5, 5, 5],
    [2, 2, 5, 5, 2, 5, 1, 1, 5, 3],
    [5, 5, 5, 5, 5, 5, 5, 1, 5, 5],
    [1, 5, 5, 5, 5, 5, 3, 5, 5, 3],
    [1, 1, 5, 5, 5, 5, 5, 5, 3, 5],
], dtype=int)

E3_IN = np.array([
    [0, 0, 5, 5, 0, 5, 5, 5, 0, 5],
    [5, 5, 0, 0, 5, 5, 5, 5, 0, 5],
    [5, 0, 5, 0, 5, 0, 5, 5, 0, 5],
    [5, 0, 5, 5, 0, 5, 5, 5, 5, 5],
    [5, 5, 5, 0, 0, 5, 5, 0, 5, 0],
    [5, 5, 0, 5, 5, 5, 5, 0, 5, 0],
    [5, 5, 0, 5, 5, 0, 5, 5, 5, 5],
    [5, 5, 5, 0, 5, 5, 5, 5, 5, 5],
    [5, 0, 5, 5, 5, 0, 5, 0, 5, 5],
    [5, 5, 0, 5, 5, 5, 5, 5, 5, 5],
], dtype=int)

E3_OUT = np.array([
    [2, 2, 5, 5, 3, 5, 5, 5, 1, 5],
    [5, 5, 1, 1, 5, 5, 5, 5, 1, 5],
    [5, 2, 5, 1, 5, 3, 5, 5, 1, 5],
    [5, 2, 5, 5, 1, 5, 5, 5, 5, 5],
    [5, 5, 5, 1, 1, 5, 5, 2, 5, 2],
    [5, 5, 2, 5, 5, 5, 5, 2, 5, 2],
    [5, 5, 2, 5, 5, 3, 5, 5, 5, 5],
    [5, 5, 5, 3, 5, 5, 5, 5, 5, 5],
    [5, 3, 5, 5, 5, 3, 5, 3, 5, 5],
    [5, 5, 3, 5, 5, 5, 5, 5, 5, 5],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 5, 5, 5, 5, 5, 0, 0, 5, 5],
    [5, 5, 5, 0, 5, 5, 0, 5, 0, 5],
    [5, 5, 0, 5, 5, 5, 5, 5, 0, 5],
    [5, 0, 0, 5, 5, 5, 5, 5, 5, 5],
    [0, 5, 5, 5, 5, 5, 0, 5, 5, 5],
    [0, 5, 5, 0, 5, 5, 0, 5, 0, 0],
    [5, 5, 0, 5, 5, 5, 5, 5, 0, 5],
    [5, 5, 0, 5, 5, 5, 5, 5, 5, 0],
    [0, 0, 5, 5, 5, 5, 0, 5, 5, 5],
    [5, 5, 5, 5, 0, 5, 0, 0, 5, 0],
], dtype=int)

T_OUT = np.array([
    [3, 5, 5, 5, 5, 5, 1, 1, 5, 5],
    [5, 5, 5, 3, 5, 5, 1, 5, 2, 5],
    [5, 5, 1, 5, 5, 5, 5, 5, 2, 5],
    [5, 1, 1, 5, 5, 5, 5, 5, 5, 5],
    [2, 5, 5, 5, 5, 5, 2, 5, 5, 5],
    [2, 5, 5, 3, 5, 5, 2, 5, 1, 1],
    [5, 5, 2, 5, 5, 5, 5, 5, 1, 5],
    [5, 5, 2, 5, 5, 5, 5, 5, 5, 3],
    [2, 2, 5, 5, 5, 5, 1, 5, 5, 5],
    [5, 5, 5, 5, 3, 5, 1, 1, 5, 3],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
	A=range;c=set();E=[c[:]for c in j]
	def F(k,W):
		if(k,W)in c or not(0<=k<10 and 0<=W<10)or j[k][W]:return[]
		c.add((k,W));return[(k,W)]+sum([F(k+c,W+l)for(c,l)in[(-1,0),(1,0),(0,-1),(0,1)]],[])
	for l in A(10):
		for J in A(10):
			if j[l][J]==0 and(l,J)not in c:
				a=F(l,J)
				for(C,e)in a:E[C][e]=abs(len(a)-4)
	return E


# --- Code Golf Solution (Compressed) ---
def q(g, n=7):
    return g * -n or [*map(lambda *r, b=5: [(b := (a * (a > 4) or [a + b // 4 - 0.25, min(a, b)][n > 3])) for a in r], *p(g, n - 1)[::-1])]


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, uniform

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

ContainerContainer = Container[Container]

ORIGIN = (0, 0)

UNITY = (1, 1)

DOWN = (1, 0)

RIGHT = (0, 1)

UP = (-1, 0)

LEFT = (0, -1)

def difference(
    a: Container,
    b: Container
) -> Container:
    """ difference """
    return type(a)(e for e in a if e not in b)

def merge(
    containers: ContainerContainer
) -> Container:
    """ merging """
    return type(containers)(e for c in containers for e in c)

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

def dneighbors(
    loc: IntegerTuple
) -> Indices:
    """ directly adjacent indices """
    return frozenset({(loc[0] - 1, loc[1]), (loc[0] + 1, loc[1]), (loc[0], loc[1] - 1), (loc[0], loc[1] + 1)})

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

def generate_e8593010(diff_lb: float, diff_ub: float) -> dict:
    a = frozenset({frozenset({ORIGIN})})
    b = frozenset({frozenset({ORIGIN, RIGHT}), frozenset({ORIGIN, DOWN})})
    c = frozenset({
    frozenset({ORIGIN, DOWN, UNITY}),
    frozenset({ORIGIN, DOWN, RIGHT}),
    frozenset({UNITY, DOWN, RIGHT}),
    frozenset({UNITY, ORIGIN, RIGHT}),
    shift(frozenset({ORIGIN, UP, DOWN}), DOWN),
    shift(frozenset({ORIGIN, LEFT, RIGHT}), RIGHT)
    })
    a, b, c = totuple(a), totuple(b), totuple(c)
    prs = [(a, 3), (b, 2), (c, 1)]
    cols = difference(interval(0, 10, 1), (1, 2, 3))
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    fgc = choice(remcols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    reminds = asindices(gi)
    nobjs = unifint(diff_lb, diff_ub, (1, ((h * w) // 2) // 2))
    maxtr = 10
    for k in range(nobjs):
        ntr = 0
        objs, col = choice(prs)
        obj = choice(objs)
        while ntr < maxtr:
            if len(reminds) == 0:
                break
            loc = choice(totuple(reminds))
            olcd = shift(obj, loc)
            if olcd.issubset(reminds):
                gi = fill(gi, fgc, olcd)
                go = fill(go, col, olcd)
                reminds = (reminds - olcd) - mapply(dneighbors, olcd)
                break
            ntr += 1
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

ONE = 1

TWO = 2

THREE = 3

F = False

T = True

def mostcolor(
    element: Element
) -> Integer:
    """ most common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return max(set(values), key=values.count)

def sizefilter(
    container: Container,
    n: Integer
) -> FrozenSet:
    """ filter items by size """
    return frozenset(item for item in container if len(item) == n)

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

def verify_e8593010(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = objects(I, T, F, T)
    x1 = sizefilter(x0, ONE)
    x2 = sizefilter(x0, TWO)
    x3 = sizefilter(x0, THREE)
    x4 = merge(x1)
    x5 = fill(I, THREE, x4)
    x6 = merge(x2)
    x7 = fill(x5, TWO, x6)
    x8 = merge(x3)
    x9 = fill(x7, ONE, x8)
    return x9


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_e8593010(inp)
        assert pred == _to_grid(expected), f"{name} failed"

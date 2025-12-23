# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "a61f2674"
SERIAL = "254"
URL    = "https://arcprize.org/play?task=a61f2674"

# --- Code Golf Concepts ---
CONCEPTS = [
    "separate_shapes",
    "count_tiles",
    "take_maximum",
    "take_minimum",
    "recoloring",
    "associate_colors_to_ranks",
    "remove_intruders",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 5, 0, 0, 0, 0, 0, 0],
    [0, 0, 5, 0, 0, 0, 5, 0, 0],
    [5, 0, 5, 0, 0, 0, 5, 0, 0],
    [5, 0, 5, 0, 0, 0, 5, 0, 0],
    [5, 0, 5, 0, 5, 0, 5, 0, 0],
    [5, 0, 5, 0, 5, 0, 5, 0, 5],
    [5, 0, 5, 0, 5, 0, 5, 0, 5],
    [5, 0, 5, 0, 5, 0, 5, 0, 5],
], dtype=int)

E1_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 2],
    [0, 0, 1, 0, 0, 0, 0, 0, 2],
    [0, 0, 1, 0, 0, 0, 0, 0, 2],
], dtype=int)

E2_IN = np.array([
    [0, 0, 0, 0, 5, 0, 0, 0, 0],
    [0, 0, 0, 0, 5, 0, 0, 0, 0],
    [5, 0, 0, 0, 5, 0, 0, 0, 0],
    [5, 0, 0, 0, 5, 0, 5, 0, 0],
    [5, 0, 0, 0, 5, 0, 5, 0, 0],
    [5, 0, 0, 0, 5, 0, 5, 0, 0],
    [5, 0, 0, 0, 5, 0, 5, 0, 0],
    [5, 0, 5, 0, 5, 0, 5, 0, 0],
    [5, 0, 5, 0, 5, 0, 5, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 2, 0, 1, 0, 0, 0, 0],
    [0, 0, 2, 0, 1, 0, 0, 0, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 5, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0],
    [0, 0, 0, 5, 0, 5, 0, 5, 0],
    [0, 0, 0, 5, 0, 5, 0, 5, 0],
    [0, 0, 0, 5, 0, 5, 0, 5, 0],
    [0, 0, 0, 5, 0, 5, 0, 5, 0],
    [0, 5, 0, 5, 0, 5, 0, 5, 0],
], dtype=int)

T_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 2, 0, 0, 0, 0, 0, 1, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j,A=range):
	c,E=len(j),len(j[0]);k=[0 for W in A(E)]
	for W in A(E):
		for l in A(c):
			if j[l][W]>0:k[W]+=1
			j[l][W]=0
	J=min([W for W in k if W>0]);W=k.index(J)
	for l in A(k[W]):j[-(l+1)][W]=2
	W=k.index(max(k))
	for l in A(k[W]):j[-(l+1)][W]=1
	return j


# --- Code Golf Solution (Compressed) ---
def q(g):
    return [[(sum(g[-sum(c) // 5]) | 5) % sum(g[8]) * x % 3 for *c, x in zip(*g, r)] for r in g]


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

def generate_a61f2674(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(2, remove(1, interval(0, 10, 1)))
    w = unifint(diff_lb, diff_ub, (5, 28))
    h = unifint(diff_lb, diff_ub, (w // 2 + 1, 30))
    bgc, fgc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    nbars = unifint(diff_lb, diff_ub, (2, w // 2))
    barlocs = []
    options = interval(0, w, 1)
    while len(options) > 0 and len(barlocs) < nbars:
        loc = choice(options)
        barlocs.append(loc)
        options = remove(loc, options)
        options = remove(loc + 1, options)
        options = remove(loc - 1, options)
    barheights = sample(interval(0, h, 1), nbars)
    for j, bh in zip(barlocs, barheights):
        gi = fill(gi, fgc, connect((0, j), (bh, j)))
        if bh == max(barheights):
            go = fill(go, 1, connect((0, j), (bh, j)))
        if bh == min(barheights):
            go = fill(go, 2, connect((0, j), (bh, j)))
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

ContainerContainer = Container[Container]

ONE = 1

TWO = 2

F = False

T = True

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

def argmin(
    container: Container,
    compfunc: Callable
) -> Any:
    """ smallest item by custom order """
    return min(container, key=compfunc, default=None)

def mostcolor(
    element: Element
) -> Integer:
    """ most common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return max(set(values), key=values.count)

def asindices(
    grid: Grid
) -> Indices:
    """ indices of all grid cells """
    return frozenset((i, j) for i in range(len(grid)) for j in range(len(grid[0])))

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

def cover(
    grid: Grid,
    patch: Patch
) -> Grid:
    """ remove object from grid """
    return fill(grid, mostcolor(grid), toindices(patch))

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_a61f2674(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = objects(I, T, F, T)
    x1 = argmax(x0, size)
    x2 = argmin(x0, size)
    x3 = merge(x0)
    x4 = cover(I, x3)
    x5 = fill(x4, ONE, x1)
    x6 = fill(x5, TWO, x2)
    return x6


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_a61f2674(inp)
        assert pred == _to_grid(expected), f"{name} failed"

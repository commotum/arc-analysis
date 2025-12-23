# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "67385a82"
SERIAL = "147"
URL    = "https://arcprize.org/play?task=67385a82"

# --- Code Golf Concepts ---
CONCEPTS = [
    "recoloring",
    "measure_area",
    "associate_colors_to_bools",
]

# --- Example Grids ---
E1_IN = np.array([
    [3, 3, 0],
    [0, 3, 0],
    [3, 0, 3],
], dtype=int)

E1_OUT = np.array([
    [8, 8, 0],
    [0, 8, 0],
    [3, 0, 3],
], dtype=int)

E2_IN = np.array([
    [0, 3, 0, 0, 0, 3],
    [0, 3, 3, 3, 0, 0],
    [0, 0, 0, 0, 3, 0],
    [0, 3, 0, 0, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [0, 8, 0, 0, 0, 3],
    [0, 8, 8, 8, 0, 0],
    [0, 0, 0, 0, 3, 0],
    [0, 3, 0, 0, 0, 0],
], dtype=int)

E3_IN = np.array([
    [3, 3, 0, 3],
    [3, 3, 0, 0],
    [3, 0, 0, 3],
    [0, 0, 3, 3],
], dtype=int)

E3_OUT = np.array([
    [8, 8, 0, 3],
    [8, 8, 0, 0],
    [8, 0, 0, 8],
    [0, 0, 8, 8],
], dtype=int)

E4_IN = np.array([
    [3, 3, 0, 0, 0, 0],
    [0, 3, 0, 0, 3, 0],
    [3, 0, 0, 0, 0, 0],
    [0, 3, 3, 0, 0, 0],
    [0, 3, 3, 0, 0, 3],
], dtype=int)

E4_OUT = np.array([
    [8, 8, 0, 0, 0, 0],
    [0, 8, 0, 0, 3, 0],
    [3, 0, 0, 0, 0, 0],
    [0, 8, 8, 0, 0, 0],
    [0, 8, 8, 0, 0, 3],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [3, 0, 3, 0, 3],
    [3, 3, 3, 0, 0],
    [0, 0, 0, 0, 3],
    [0, 3, 3, 0, 0],
    [0, 3, 3, 0, 0],
], dtype=int)

T_OUT = np.array([
    [8, 0, 8, 0, 3],
    [8, 8, 8, 0, 0],
    [0, 0, 0, 0, 3],
    [0, 8, 8, 0, 0],
    [0, 8, 8, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
	A=[k[:]for k in j];c,E=len(j),len(j[0])
	for k in range(c):
		for W in range(E):
			if j[k][W]==3:
				for(l,J)in[(0,1),(1,0),(0,-1),(-1,0)]:
					if 0<=k+l<c and 0<=W+J<E and j[k+l][W+J]==3:A[k][W]=8;break
	return A


# --- Code Golf Solution (Compressed) ---
def q(i, *w):
    return i * 0 != 0 and [*map(p, i, [i * 2] + i, i[1:] + [i * 2], *w)] or (3 in w) + 7 & i * 9


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, sample, uniform

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

def toindices(
    patch: Patch
) -> Indices:
    """ indices of object cells """
    if len(patch) == 0:
        return frozenset()
    if isinstance(next(iter(patch))[1], tuple):
        return frozenset(index for value, index in patch)
    return patch

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

def generate_67385a82(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(0, remove(8, interval(0, 10, 1)))
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    col = choice(cols)
    gi = canvas(0, (h, w))
    inds = totuple(asindices(gi))
    ncd = unifint(diff_lb, diff_ub, (0, len(inds) // 2))
    nc = choice((ncd, len(inds) - ncd))
    nc = min(max(1, nc), len(inds) - 1)
    locs = sample(inds, nc)
    gi = fill(gi, col, locs)
    objs = objects(gi, T, F, F)
    rems = toindices(merge(sizefilter(colorfilter(objs, col), 1)))
    blues = difference(ofcolor(gi, col), rems)
    go = fill(gi, 8, blues)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
IntegerSet = FrozenSet[Integer]

ZERO = 0

ONE = 1

EIGHT = 8

def first(
    container: Container
) -> Any:
    """ first item of container """
    return next(iter(container))

def other(
    container: Container,
    value: Any
) -> Any:
    """ other value in the container """
    return first(remove(value, container))

def palette(
    element: Element
) -> IntegerSet:
    """ colors occurring in object or grid """
    if isinstance(element, tuple):
        return frozenset({v for r in element for v in r})
    return frozenset({v for v, _ in element})

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_67385a82(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = objects(I, T, F, F)
    x1 = palette(I)
    x2 = other(x1, ZERO)
    x3 = colorfilter(x0, x2)
    x4 = sizefilter(x3, ONE)
    x5 = difference(x3, x4)
    x6 = merge(x5)
    x7 = fill(I, EIGHT, x6)
    return x7


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("E4", E4_IN, E4_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_67385a82(inp)
        assert pred == _to_grid(expected), f"{name} failed"

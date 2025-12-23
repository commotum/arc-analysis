# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "b6afb2da"
SERIAL = "283"
URL    = "https://arcprize.org/play?task=b6afb2da"

# --- Code Golf Concepts ---
CONCEPTS = [
    "recoloring",
    "replace_pattern",
    "rectangle_guessing",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 5, 5, 5, 5, 0, 0, 0, 0, 0],
    [0, 5, 5, 5, 5, 0, 0, 0, 0, 0],
    [0, 5, 5, 5, 5, 0, 0, 0, 0, 0],
    [0, 5, 5, 5, 5, 0, 5, 5, 5, 5],
    [0, 0, 0, 0, 0, 0, 5, 5, 5, 5],
    [0, 0, 0, 0, 0, 0, 5, 5, 5, 5],
    [0, 0, 0, 0, 0, 0, 5, 5, 5, 5],
    [0, 0, 0, 0, 0, 0, 5, 5, 5, 5],
], dtype=int)

E1_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 4, 4, 1, 0, 0, 0, 0, 0],
    [0, 4, 2, 2, 4, 0, 0, 0, 0, 0],
    [0, 4, 2, 2, 4, 0, 0, 0, 0, 0],
    [0, 1, 4, 4, 1, 0, 1, 4, 4, 1],
    [0, 0, 0, 0, 0, 0, 4, 2, 2, 4],
    [0, 0, 0, 0, 0, 0, 4, 2, 2, 4],
    [0, 0, 0, 0, 0, 0, 4, 2, 2, 4],
    [0, 0, 0, 0, 0, 0, 1, 4, 4, 1],
], dtype=int)

E2_IN = np.array([
    [5, 5, 5, 5, 5, 5, 0, 0, 0, 0],
    [5, 5, 5, 5, 5, 5, 0, 0, 0, 0],
    [5, 5, 5, 5, 5, 5, 0, 0, 0, 0],
    [5, 5, 5, 5, 5, 5, 0, 0, 0, 0],
    [5, 5, 5, 5, 5, 5, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 0, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 0, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 0, 5, 5, 5, 5, 5, 5],
], dtype=int)

E2_OUT = np.array([
    [1, 4, 4, 4, 4, 1, 0, 0, 0, 0],
    [4, 2, 2, 2, 2, 4, 0, 0, 0, 0],
    [4, 2, 2, 2, 2, 4, 0, 0, 0, 0],
    [4, 2, 2, 2, 2, 4, 0, 0, 0, 0],
    [1, 4, 4, 4, 4, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 4, 4, 4, 4, 1],
    [0, 0, 0, 0, 4, 2, 2, 2, 2, 4],
    [0, 0, 0, 0, 4, 2, 2, 2, 2, 4],
    [0, 0, 0, 0, 1, 4, 4, 4, 4, 1],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 5, 5, 5, 5, 0, 0, 0, 0, 0],
    [0, 5, 5, 5, 5, 0, 0, 0, 0, 0],
    [0, 5, 5, 5, 5, 0, 0, 0, 0, 0],
    [0, 5, 5, 5, 5, 0, 0, 0, 0, 0],
    [0, 5, 5, 5, 5, 0, 0, 0, 0, 0],
    [0, 5, 5, 5, 5, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 0, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 0, 5, 5, 5, 5, 5, 5],
], dtype=int)

T_OUT = np.array([
    [0, 1, 4, 4, 1, 0, 0, 0, 0, 0],
    [0, 4, 2, 2, 4, 0, 0, 0, 0, 0],
    [0, 4, 2, 2, 4, 0, 0, 0, 0, 0],
    [0, 4, 2, 2, 4, 0, 0, 0, 0, 0],
    [0, 4, 2, 2, 4, 0, 0, 0, 0, 0],
    [0, 1, 4, 4, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 4, 4, 4, 4, 1],
    [0, 0, 0, 0, 4, 2, 2, 2, 2, 4],
    [0, 0, 0, 0, 1, 4, 4, 4, 4, 1],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def f(j,p,A,c,E,k):
 for W in range(A,E+1):
  for l in range(p,c+1):j[W][l]=k
def z(j,p,A,c,E):f(j,p,A,c,E,4);f(j,p+1,A+1,c-1,E-1,2);j[A][p]=j[A][c]=j[E][p]=j[E][c]=1
def p(j):
 J,a=len(j),len(j[0])
 for C in range(J*a):
  l,W=C%a,C//a
  if j[W][l]==5:
   c,E=l,W
   while c<a-1 and j[E][c+1]==5:c+=1
   while E<J-1 and j[E+1][c]==5:E+=1
   z(j,l,W,c,E)
 return j


# --- Code Golf Solution (Compressed) ---
def q(i, *w):
    return i * 0 != 0 and [*map(p, i, [i] + i, i[1:] + [i], *w)] or -i % 8 * w.count(5) % 5


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

def urcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of upper right corner """
    return tuple(map(lambda ix: {0: min, 1: max}[ix[0]](ix[1]), enumerate(zip(*toindices(patch)))))

def llcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of lower left corner """
    return tuple(map(lambda ix: {0: max, 1: min}[ix[0]](ix[1]), enumerate(zip(*toindices(patch)))))

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

def corners(
    patch: Patch
) -> Indices:
    """ indices of corners """
    return frozenset({ulcorner(patch), urcorner(patch), llcorner(patch), lrcorner(patch)})

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

def generate_b6afb2da(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2, 4))    
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    num = unifint(diff_lb, diff_ub, (1, 9))
    indss = asindices(gi)
    maxtrials = 4 * num
    tr = 0
    succ = 0
    while succ < num and tr <= maxtrials:
        if len(remcols) == 0 or len(indss) == 0:
            break
        oh = randint(3, 7)
        ow = randint(3, 7)
        subs = totuple(sfilter(indss, lambda ij: ij[0] < h - oh and ij[1] < w - ow))
        if len(subs) == 0:
            tr += 1
            continue
        loci, locj = choice(subs)
        obj = frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)})
        bd = backdrop(obj)
        col = choice(remcols)
        if bd.issubset(indss):
            remcols = remove(col, remcols)
            gi = fill(gi, col, bd)
            go = fill(go, 2, bd)
            go = fill(go, 4, box(bd))
            go = fill(go, 1, corners(bd))
            succ += 1
            indss = indss - bd
        tr += 1
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

ContainerContainer = Container[Container]

ONE = 1

TWO = 2

FOUR = 4

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

def merge(
    containers: ContainerContainer
) -> Container:
    """ merging """
    return type(containers)(e for c in containers for e in c)

def extract(
    container: Container,
    condition: Callable
) -> Any:
    """ first element of container that satisfies condition """
    return next(e for e in container if condition(e))

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

def verify_b6afb2da(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = objects(I, T, F, F)
    x1 = fork(equality, toindices, backdrop)
    x2 = compose(flip, x1)
    x3 = extract(x0, x2)
    x4 = color(x3)
    x5 = matcher(color, x4)
    x6 = compose(flip, x5)
    x7 = sfilter(x0, x6)
    x8 = merge(x7)
    x9 = fill(I, TWO, x8)
    x10 = mapply(box, x7)
    x11 = fill(x9, FOUR, x10)
    x12 = mapply(corners, x7)
    x13 = fill(x11, ONE, x12)
    return x13


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_b6afb2da(inp)
        assert pred == _to_grid(expected), f"{name} failed"

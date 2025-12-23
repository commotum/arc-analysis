# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "d90796e8"
SERIAL = "344"
URL    = "https://arcprize.org/play?task=d90796e8"

# --- Code Golf Concepts ---
CONCEPTS = [
    "replace_pattern",
]

# --- Example Grids ---
E1_IN = np.array([
    [3, 2, 0],
    [0, 0, 0],
    [0, 5, 0],
], dtype=int)

E1_OUT = np.array([
    [8, 0, 0],
    [0, 0, 0],
    [0, 5, 0],
], dtype=int)

E2_IN = np.array([
    [5, 0, 0, 0, 0, 0],
    [0, 0, 3, 2, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 3, 0, 0, 0, 2],
    [0, 2, 0, 0, 0, 0],
    [5, 0, 0, 3, 0, 0],
    [0, 0, 0, 0, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [5, 0, 0, 0, 0, 0],
    [0, 0, 8, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 8, 0, 0, 0, 2],
    [0, 0, 0, 0, 0, 0],
    [5, 0, 0, 3, 0, 0],
    [0, 0, 0, 0, 0, 0],
], dtype=int)

E3_IN = np.array([
    [0, 0, 0, 0, 0, 2, 0],
    [3, 0, 0, 0, 0, 0, 3],
    [5, 0, 2, 3, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 2, 0],
    [3, 2, 0, 0, 0, 3, 0],
    [0, 0, 0, 5, 0, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [0, 0, 0, 0, 0, 2, 0],
    [3, 0, 0, 0, 0, 0, 3],
    [5, 0, 0, 8, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [8, 0, 0, 0, 0, 8, 0],
    [0, 0, 0, 5, 0, 0, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 0, 0, 2, 0, 0, 0, 5],
    [0, 2, 0, 0, 0, 0, 3, 2, 0],
    [0, 3, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 5, 0, 0, 0, 2],
    [5, 0, 0, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 3, 0],
    [5, 3, 0, 0, 0, 5, 0, 2, 0],
], dtype=int)

T_OUT = np.array([
    [0, 0, 0, 0, 2, 0, 0, 0, 5],
    [0, 0, 0, 0, 0, 0, 8, 0, 0],
    [0, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 5, 0, 0, 0, 2],
    [5, 0, 0, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 8, 0],
    [5, 3, 0, 0, 0, 5, 0, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j,A=enumerate):
 for c,E in A(j):
  for k,W in A(E):
   for l,J in(c+1,k),(c-1,k),(c,k+1),(c,k-1):
    if W==2 and 0<=l<len(j)and 0<=J<len(E)and j[l][J]==3:j[c][k]=0;j[l][J]=8
 return j


# --- Code Golf Solution (Compressed) ---
def q(i, *w):
    return i * 0 != 0 and [*map(p, i, [i * 4] + i, i[1:] + [i * 4], *w)] or (i ^ 1 in w) + 7 & i * 9


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

def generate_d90796e8(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (8, 2, 3))
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc, noisec = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    nocc = unifint(diff_lb, diff_ub, (1, (h * w) // 3))
    inds = asindices(gi)
    locs = sample(totuple(inds), nocc)
    obj = frozenset({(choice((noisec, 2, 3)), ij) for ij in locs})
    gi = paint(gi, obj)
    fixloc = choice(totuple(inds))
    fixloc2 = choice(totuple(dneighbors(fixloc) & inds))
    gi = fill(gi, 2, {fixloc})
    gi = fill(gi, 3, {fixloc2})
    go = tuple(e for e in gi)
    reds = ofcolor(gi, 2)
    greens = ofcolor(gi, 3)
    tocover = set()
    tolblue = set()
    for r in reds:
        inters = dneighbors(r) & greens
        if len(inters) > 0:
            tocover.add(r)
            tolblue = tolblue | inters
    go = fill(go, bgc, tocover)
    go = fill(go, 8, tolblue)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Element = Union[Object, Grid]

TWO = 2

THREE = 3

EIGHT = 8

def intersection(
    a: FrozenSet,
    b: FrozenSet
) -> FrozenSet:
    """ returns the intersection of two containers """
    return a & b

def size(
    container: Container
) -> Integer:
    """ cardinality """
    return len(container)

def positive(
    x: Integer
) -> Boolean:
    """ positive """
    return x > 0

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

def chain(
    h: Callable,
    g: Callable,
    f: Callable
) -> Callable:
    """ function composition with three functions """
    return lambda x: h(g(f(x)))

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

def mostcolor(
    element: Element
) -> Integer:
    """ most common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return max(set(values), key=values.count)

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

def verify_d90796e8(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = ofcolor(I, TWO)
    x1 = ofcolor(I, THREE)
    x2 = compose(positive, size)
    x3 = lbind(intersection, x1)
    x4 = chain(x2, x3, dneighbors)
    x5 = compose(positive, size)
    x6 = lbind(intersection, x0)
    x7 = chain(x5, x6, dneighbors)
    x8 = sfilter(x0, x4)
    x9 = sfilter(x1, x7)
    x10 = cover(I, x8)
    x11 = fill(x10, EIGHT, x9)
    return x11


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_d90796e8(inp)
        assert pred == _to_grid(expected), f"{name} failed"

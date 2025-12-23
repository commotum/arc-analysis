# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "a5f85a15"
SERIAL = "252"
URL    = "https://arcprize.org/play?task=a5f85a15"

# --- Code Golf Concepts ---
CONCEPTS = [
    "recoloring",
    "pattern_modification",
    "pairwise_analogy",
]

# --- Example Grids ---
E1_IN = np.array([
    [2, 0, 0],
    [0, 2, 0],
    [0, 0, 2],
], dtype=int)

E1_OUT = np.array([
    [2, 0, 0],
    [0, 4, 0],
    [0, 0, 2],
], dtype=int)

E2_IN = np.array([
    [0, 0, 9, 0, 0, 0, 0, 0],
    [0, 0, 0, 9, 0, 0, 0, 0],
    [0, 0, 0, 0, 9, 0, 0, 0],
    [0, 0, 0, 0, 0, 9, 0, 0],
    [9, 0, 0, 0, 0, 0, 9, 0],
    [0, 9, 0, 0, 0, 0, 0, 9],
    [0, 0, 9, 0, 0, 0, 0, 0],
    [0, 0, 0, 9, 0, 0, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [0, 0, 9, 0, 0, 0, 0, 0],
    [0, 0, 0, 4, 0, 0, 0, 0],
    [0, 0, 0, 0, 9, 0, 0, 0],
    [0, 0, 0, 0, 0, 4, 0, 0],
    [9, 0, 0, 0, 0, 0, 9, 0],
    [0, 4, 0, 0, 0, 0, 0, 4],
    [0, 0, 9, 0, 0, 0, 0, 0],
    [0, 0, 0, 4, 0, 0, 0, 0],
], dtype=int)

E3_IN = np.array([
    [0, 0, 3, 0, 0, 0],
    [0, 0, 0, 3, 0, 0],
    [0, 0, 0, 0, 3, 0],
    [3, 0, 0, 0, 0, 3],
    [0, 3, 0, 0, 0, 0],
    [0, 0, 3, 0, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [0, 0, 3, 0, 0, 0],
    [0, 0, 0, 4, 0, 0],
    [0, 0, 0, 0, 3, 0],
    [3, 0, 0, 0, 0, 4],
    [0, 4, 0, 0, 0, 0],
    [0, 0, 3, 0, 0, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0],
    [6, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0],
    [0, 6, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0],
    [0, 0, 6, 0, 0, 0, 0, 6, 0, 0, 0, 0],
    [0, 0, 0, 6, 0, 0, 0, 0, 6, 0, 0, 0],
    [0, 0, 0, 0, 6, 0, 0, 0, 0, 6, 0, 0],
    [0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 6, 0],
    [0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 6],
    [6, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0],
    [0, 6, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0],
    [0, 0, 6, 0, 0, 0, 0, 0, 0, 6, 0, 0],
    [0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 6, 0],
], dtype=int)

T_OUT = np.array([
    [0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0],
    [6, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0],
    [0, 4, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0],
    [0, 0, 6, 0, 0, 0, 0, 4, 0, 0, 0, 0],
    [0, 0, 0, 4, 0, 0, 0, 0, 6, 0, 0, 0],
    [0, 0, 0, 0, 6, 0, 0, 0, 0, 4, 0, 0],
    [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 6, 0],
    [0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 4],
    [6, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
    [0, 4, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0],
    [0, 0, 6, 0, 0, 0, 0, 0, 0, 4, 0, 0],
    [0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 6, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j,A=range):
 c=len(j)
 for E in A(c):
  for k,W in zip(A(1,c,2),A(E+1,c,2)):
   if j[0][E]:j[k][W]=4
   if j[E][0]:j[W][k]=4
 return j


# --- Code Golf Solution (Compressed) ---
def q(g, v=0):
    return g * 0 != 0 and [*map(p, g, [-1, 4] * 9)] or -g // v & v


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

def order(
    container: Container,
    compfunc: Callable
) -> Tuple:
    """ order container by custom key """
    return tuple(sorted(container, key=compfunc))

def toivec(
    i: Integer
) -> IntegerTuple:
    """ vector pointing vertically """
    return (i, 0)

def tojvec(
    j: Integer
) -> IntegerTuple:
    """ vector pointing horizontally """
    return (0, j)

def first(
    container: Container
) -> Any:
    """ first item of container """
    return next(iter(container))

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

def shoot(
    start: IntegerTuple,
    direction: IntegerTuple
) -> Indices:
    """ line from starting point and direction """
    return connect(start, (start[0] + 42 * direction[0], start[1] + 42 * direction[1]))

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

def generate_a5f85a15(diff_lb: float, diff_ub: float) -> dict:
    colopts = remove(4, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    startlocs = apply(toivec, interval(h - 1, 0, -1)) + apply(tojvec, interval(0, w, 1))
    cands = interval(0, h + w - 1, 1)
    num = unifint(diff_lb, diff_ub, (1, (h + w - 1) // 3))
    locs = []
    for k in range(num):
        if len(cands) == 0:
            break
        loc = choice(cands)
        locs.append(loc)
        cands = remove(loc, cands)
        cands = remove(loc - 1, cands)
        cands = remove(loc + 1, cands)
    locs = set([startlocs[loc] for loc in locs])
    bgc, fgc = sample(colopts, 2)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    for loc in locs:
        ln = order(shoot(loc, (1, 1)), first)
        gi = fill(gi, fgc, ln)
        go = fill(go, fgc, ln)
        go = fill(go, 4, ln[1::2])
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Numerical = Union[Integer, IntegerTuple]

Element = Union[Object, Grid]

ContainerContainer = Container[Container]

ZERO = 0

FOUR = 4

ORIGIN = (0, 0)

UNITY = (1, 1)

def identity(
    x: Any
) -> Any:
    """ identity function """
    return x

def double(
    n: Numerical
) -> Numerical:
    """ scaling by two """
    return n * 2 if isinstance(n, int) else (n[0] * 2, n[1] * 2)

def contained(
    value: Any,
    container: Container
) -> Boolean:
    """ element of """
    return value in container

def merge(
    containers: ContainerContainer
) -> Container:
    """ merging """
    return type(containers)(e for c in containers for e in c)

def increment(
    x: Numerical
) -> Numerical:
    """ incrementing """
    return x + 1 if isinstance(x, int) else (x[0] + 1, x[1] + 1)

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

def mapply(
    function: Callable,
    container: ContainerContainer
) -> FrozenSet:
    """ apply and merge """
    return merge(apply(function, container))

def leastcolor(
    element: Element
) -> Integer:
    """ least common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return min(set(values), key=values.count)

def ofcolor(
    grid: Grid,
    value: Integer
) -> Indices:
    """ indices of all grid cells with value """
    return frozenset((i, j) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value)

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

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_a5f85a15(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = leastcolor(I)
    x1 = ofcolor(I, x0)
    x2 = compose(increment, double)
    x3 = shoot(ORIGIN, UNITY)
    x4 = apply(x2, x3)
    x5 = order(x4, identity)
    x6 = lbind(contained, ZERO)
    x7 = sfilter(x1, x6)
    x8 = lbind(shift, x5)
    x9 = mapply(x8, x7)
    x10 = fill(I, FOUR, x9)
    return x10


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_a5f85a15(inp)
        assert pred == _to_grid(expected), f"{name} failed"

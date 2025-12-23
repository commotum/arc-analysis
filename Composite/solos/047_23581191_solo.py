# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "23581191"
SERIAL = "047"
URL    = "https://arcprize.org/play?task=23581191"

# --- Code Golf Concepts ---
CONCEPTS = [
    "draw_line_from_point",
    "pattern_intersection",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 8, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 7, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [0, 0, 8, 0, 0, 0, 7, 0, 0],
    [0, 0, 8, 0, 0, 0, 7, 0, 0],
    [8, 8, 8, 8, 8, 8, 2, 8, 8],
    [0, 0, 8, 0, 0, 0, 7, 0, 0],
    [0, 0, 8, 0, 0, 0, 7, 0, 0],
    [0, 0, 8, 0, 0, 0, 7, 0, 0],
    [7, 7, 2, 7, 7, 7, 7, 7, 7],
    [0, 0, 8, 0, 0, 0, 7, 0, 0],
    [0, 0, 8, 0, 0, 0, 7, 0, 0],
], dtype=int)

E2_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 7, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [0, 0, 0, 8, 0, 0, 7, 0, 0],
    [8, 8, 8, 8, 8, 8, 2, 8, 8],
    [0, 0, 0, 8, 0, 0, 7, 0, 0],
    [0, 0, 0, 8, 0, 0, 7, 0, 0],
    [0, 0, 0, 8, 0, 0, 7, 0, 0],
    [0, 0, 0, 8, 0, 0, 7, 0, 0],
    [0, 0, 0, 8, 0, 0, 7, 0, 0],
    [7, 7, 7, 2, 7, 7, 7, 7, 7],
    [0, 0, 0, 8, 0, 0, 7, 0, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 7, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

T_OUT = np.array([
    [0, 7, 0, 0, 8, 0, 0, 0, 0],
    [8, 2, 8, 8, 8, 8, 8, 8, 8],
    [0, 7, 0, 0, 8, 0, 0, 0, 0],
    [0, 7, 0, 0, 8, 0, 0, 0, 0],
    [0, 7, 0, 0, 8, 0, 0, 0, 0],
    [0, 7, 0, 0, 8, 0, 0, 0, 0],
    [7, 7, 7, 7, 2, 7, 7, 7, 7],
    [0, 7, 0, 0, 8, 0, 0, 0, 0],
    [0, 7, 0, 0, 8, 0, 0, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
	A=range;c=[[0]*9 for c in A(9)];E=[(c,E,j[c][E])for c in A(9)for E in A(9)if j[c][E]]
	for(k,W,l)in E:
		for J in range(9):c[k][J]=c[J][W]=l
	c[E[0][0]][E[1][1]]=c[E[1][0]][E[0][1]]=2;return c


# --- Code Golf Solution (Compressed) ---
def q(m):
    return [[sum({*r + c}) % 13 for *c, in zip(*m)] for r in m]


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

def combine(
    a: Container,
    b: Container
) -> Container:
    """ union """
    return type(a)((*a, *b))

def intersection(
    a: FrozenSet,
    b: FrozenSet
) -> FrozenSet:
    """ returns the intersection of two containers """
    return a & b

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

def first(
    container: Container
) -> Any:
    """ first item of container """
    return next(iter(container))

def last(
    container: Container
) -> Any:
    """ last item of container """
    return max(enumerate(container))[1]

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

def generate_23581191(diff_lb: float, diff_ub: float) -> dict:
    dim_bounds = (3, 30)
    colopts = remove(2, interval(0, 10, 1))
    f = fork(combine, hfrontier, vfrontier)
    h = unifint(diff_lb, diff_ub, dim_bounds)
    w = unifint(diff_lb, diff_ub, dim_bounds)
    bgcol = choice(colopts)
    remcols = remove(bgcol, colopts)
    c = canvas(bgcol, (h, w))
    inds = totuple(asindices(c))
    acol = choice(remcols)
    bcol = choice(remove(acol, remcols))
    card_bounds = (1, (h * w) // 4)
    na = unifint(diff_lb, diff_ub, card_bounds)
    nb = unifint(diff_lb, diff_ub, card_bounds)
    a = sample(inds, na)
    b = sample(difference(inds, a), nb)
    gi = fill(c, acol, a)
    gi = fill(gi, bcol, b)
    fa = apply(first, a)
    la = apply(last, a)
    fb = apply(first, b)
    lb = apply(last, b)
    alins = sfilter(inds, lambda ij: first(ij) in fa or last(ij) in la)
    blins = sfilter(inds, lambda ij: first(ij) in fb or last(ij) in lb)
    go = fill(c, acol, alins)
    go = fill(go, bcol, blins)
    go = fill(go, 2, intersection(set(alins), set(blins)))
    go = fill(go, acol, a)
    go = fill(go, bcol, b)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
IntegerSet = FrozenSet[Integer]

Element = Union[Object, Grid]

ContainerContainer = Container[Container]

TWO = 2

def merge(
    containers: ContainerContainer
) -> Container:
    """ merging """
    return type(containers)(e for c in containers for e in c)

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

def mostcolor(
    element: Element
) -> Integer:
    """ most common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return max(set(values), key=values.count)

def ofcolor(
    grid: Grid,
    value: Integer
) -> Indices:
    """ indices of all grid cells with value """
    return frozenset((i, j) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value)

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

def verify_23581191(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = mostcolor(I)
    x1 = palette(I)
    x2 = remove(x0, x1)
    x3 = totuple(x2)
    x4 = fork(combine, hfrontier, vfrontier)
    x5 = lbind(mapply, x4)
    x6 = lbind(ofcolor, I)
    x7 = compose(x5, x6)
    x8 = first(x3)
    x9 = last(x3)
    x10 = x7(x8)
    x11 = x7(x9)
    x12 = ofcolor(I, x0)
    x13 = intersection(x12, x10)
    x14 = intersection(x12, x11)
    x15 = intersection(x10, x11)
    x16 = intersection(x12, x15)
    x17 = fill(I, x8, x13)
    x18 = fill(x17, x9, x14)
    x19 = fill(x18, TWO, x16)
    return x19


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_23581191(inp)
        assert pred == _to_grid(expected), f"{name} failed"

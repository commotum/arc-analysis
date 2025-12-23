# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "2bee17df"
SERIAL = "063"
URL    = "https://arcprize.org/play?task=2bee17df"

# --- Code Golf Concepts ---
CONCEPTS = [
    "draw_line_from_border",
    "count_tiles",
    "take_maximum",
]

# --- Example Grids ---
E1_IN = np.array([
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [8, 0, 0, 0, 2, 2, 0, 2, 2, 2, 2, 2],
    [8, 0, 0, 0, 0, 2, 0, 0, 2, 2, 0, 2],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
    [8, 8, 0, 0, 0, 0, 8, 8, 0, 0, 0, 8],
    [8, 8, 8, 0, 0, 8, 8, 8, 0, 0, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
], dtype=int)

E1_OUT = np.array([
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [8, 0, 0, 3, 2, 2, 0, 2, 2, 2, 2, 2],
    [8, 0, 0, 3, 0, 2, 0, 0, 2, 2, 0, 2],
    [8, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2],
    [8, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2],
    [8, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2],
    [8, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2],
    [8, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 8],
    [8, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 8],
    [8, 8, 0, 3, 0, 0, 8, 8, 0, 0, 0, 8],
    [8, 8, 8, 3, 0, 8, 8, 8, 0, 0, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
], dtype=int)

E2_IN = np.array([
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [2, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8],
    [2, 2, 0, 0, 0, 0, 0, 8, 8, 0, 0, 8],
    [2, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 8],
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
    [2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 8],
    [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
    [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2],
    [2, 2, 0, 2, 0, 0, 2, 0, 0, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
], dtype=int)

E2_OUT = np.array([
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [2, 0, 0, 0, 3, 3, 8, 8, 8, 8, 8, 8],
    [2, 2, 0, 0, 3, 3, 0, 8, 8, 0, 0, 8],
    [2, 0, 0, 0, 3, 3, 0, 8, 0, 0, 0, 8],
    [2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 8],
    [2, 2, 2, 0, 3, 3, 0, 0, 0, 0, 0, 8],
    [2, 2, 0, 0, 3, 3, 0, 0, 0, 0, 0, 8],
    [2, 2, 0, 0, 3, 3, 0, 0, 0, 0, 0, 8],
    [2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 8],
    [2, 0, 0, 0, 3, 3, 0, 0, 0, 0, 2, 2],
    [2, 2, 0, 2, 3, 3, 2, 0, 0, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
], dtype=int)

E3_IN = np.array([
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 2],
    [8, 8, 8, 0, 8, 8, 0, 8, 0, 2],
    [8, 8, 0, 0, 8, 0, 0, 0, 0, 2],
    [8, 8, 0, 0, 0, 0, 0, 0, 2, 2],
    [8, 0, 0, 0, 0, 0, 0, 0, 2, 2],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [8, 0, 0, 0, 0, 0, 2, 2, 0, 2],
    [8, 2, 0, 0, 0, 2, 2, 2, 2, 2],
    [8, 2, 2, 2, 2, 2, 2, 2, 2, 2],
], dtype=int)

E3_OUT = np.array([
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 2],
    [8, 8, 8, 3, 8, 8, 0, 8, 0, 2],
    [8, 8, 0, 3, 8, 0, 0, 0, 0, 2],
    [8, 8, 0, 3, 0, 0, 0, 0, 2, 2],
    [8, 0, 0, 3, 0, 0, 0, 0, 2, 2],
    [8, 3, 3, 3, 3, 3, 3, 3, 3, 2],
    [8, 3, 3, 3, 3, 3, 3, 3, 3, 2],
    [8, 0, 0, 3, 0, 0, 2, 2, 0, 2],
    [8, 2, 0, 3, 0, 2, 2, 2, 2, 2],
    [8, 2, 2, 2, 2, 2, 2, 2, 2, 2],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 0, 0, 8, 8, 8, 0, 0, 8, 2, 2],
    [8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2],
    [8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2],
    [8, 8, 0, 2, 0, 2, 2, 0, 0, 0, 0, 2, 2, 2],
    [8, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
], dtype=int)

T_OUT = np.array([
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 3, 0, 8, 8, 8, 3, 3, 8, 2, 2],
    [8, 8, 8, 0, 3, 0, 0, 0, 0, 3, 3, 0, 0, 2],
    [8, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2],
    [8, 8, 8, 0, 3, 0, 0, 0, 0, 3, 3, 0, 2, 2],
    [8, 8, 0, 0, 3, 0, 0, 0, 0, 3, 3, 2, 2, 2],
    [8, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2],
    [8, 8, 0, 0, 3, 0, 0, 0, 0, 3, 3, 0, 0, 2],
    [8, 8, 0, 0, 3, 0, 0, 0, 0, 3, 3, 0, 0, 2],
    [8, 8, 0, 0, 3, 0, 0, 0, 0, 3, 3, 0, 0, 2],
    [8, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2],
    [8, 8, 0, 0, 3, 0, 0, 0, 0, 3, 3, 0, 2, 2],
    [8, 8, 0, 2, 3, 2, 2, 0, 0, 3, 3, 2, 2, 2],
    [8, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
 A=range
 c=len(j)
 E=[o[:]for o in j]
 for k in range(c):
  if j[1][k]==0 and j[c-2][k]==0 and sum(j[W][k]for W in A(1,c-1))==0:
   for W in A(1,c-1):E[W][k]=3
 for W in range(c):
  if j[W][1]==0 and j[W][c-2]==0 and sum(j[W][k]for k in A(1,c-1))==0:
   for k in A(1,c-1):
    if E[W][k]==0:E[W][k]=3
 return E


# --- Code Golf Solution (Compressed) ---
def q(g):
    return [[x | 3 >> x + sum(r[1:-any(c)]) for _, *c, _, x in zip(*g, r)] for r in g]


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, randint, sample, uniform

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Objects = FrozenSet[Object]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

Piece = Union[Grid, Patch]

ContainerContainer = Container[Container]

def merge(
    containers: ContainerContainer
) -> Container:
    """ merging """
    return type(containers)(e for c in containers for e in c)

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

def astuple(
    a: Integer,
    b: Integer
) -> IntegerTuple:
    """ constructs a tuple """
    return (a, b)

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

def dmirror(
    piece: Piece
) -> Piece:
    """ mirroring along diagonal """
    if isinstance(piece, tuple):
        return tuple(zip(*piece))
    a, b = ulcorner(piece)
    if isinstance(next(iter(piece))[1], tuple):
        return frozenset((v, (j - b + a, i - a + b)) for v, (i, j) in piece)
    return frozenset((j - b + a, i - a + b) for i, j in piece)

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

def trim(
    grid: Grid
) -> Grid:
    """ trim border of grid """
    return tuple(r[1:-1] for r in grid[1:-1])

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

def frontiers(
    grid: Grid
) -> Objects:
    """ set of frontiers """
    h, w = len(grid), len(grid[0])
    row_indices = tuple(i for i, r in enumerate(grid) if len(set(r)) == 1)
    column_indices = tuple(j for j, c in enumerate(dmirror(grid)) if len(set(c)) == 1)
    hfrontiers = frozenset({frozenset({(grid[i][j], (i, j)) for j in range(w)}) for i in row_indices})
    vfrontiers = frozenset({frozenset({(grid[i][j], (i, j)) for i in range(h)}) for j in column_indices})
    return hfrontiers | vfrontiers

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

def generate_2bee17df(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(3, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (7, 30))
    w = unifint(diff_lb, diff_ub, (7, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    c = canvas(bgc, (h, w))
    indord1 = apply(tojvec, interval(0, w, 1))
    indord2 = apply(rbind(astuple, w - 1), interval(1, h - 1, 1))
    indord3 = apply(lbind(astuple, h - 1), interval(w - 1, 0, -1))
    indord4 = apply(toivec, interval(h - 1, 0, -1))
    indord = indord1 + indord2 + indord3 + indord4
    k = len(indord)
    sp = randint(0, k)
    arr = indord[sp:] + indord[:sp]
    ep = randint(k // 2 - 3, k // 2 + 1)
    a = arr[:ep]
    b = arr[ep:]
    cola = choice(remcols)
    remcols = remove(cola, remcols)
    colb = choice(remcols)
    gi = fill(c, cola, a)
    gi = fill(gi, colb, b)
    nr = unifint(diff_lb, diff_ub, (1, min(4, min(h, w) // 2)))
    for kk in range(nr):
        ring = box(frozenset({(1 + kk, 1 + kk), (h - 1 - kk, w - 1 - kk)}))
        for br in (cola, colb):
            blacks = ofcolor(gi, br)
            bcands = totuple(ring & ofcolor(gi, bgc) & mapply(dneighbors, ofcolor(gi, br)))
            jj = len(bcands)
            jj2 = randint(max(0, jj // 2 - 2), min(jj, jj // 2 + 1))
            ss = sample(bcands, jj2)
            gi = fill(gi, br, ss)
    res = shift(merge(frontiers(trim(gi))), (1, 1))
    go = fill(gi, 3, res)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Element = Union[Object, Grid]

ONE = 1

THREE = 3

UNITY = (1, 1)

def identity(
    x: Any
) -> Any:
    """ identity function """
    return x

def combine(
    a: Container,
    b: Container
) -> Container:
    """ union """
    return type(a)((*a, *b))

def dedupe(
    iterable: Tuple
) -> Tuple:
    """ remove duplicates """
    return tuple(e for i, e in enumerate(iterable) if iterable.index(e) == i)

def repeat(
    item: Any,
    num: Integer
) -> Tuple:
    """ repetition of item within vector """
    return tuple(item for i in range(num))

def size(
    container: Container
) -> Integer:
    """ cardinality """
    return len(container)

def initset(
    value: Any
) -> FrozenSet:
    """ initialize container """
    return frozenset({value})

def first(
    container: Container
) -> Any:
    """ first item of container """
    return next(iter(container))

def branch(
    condition: Boolean,
    if_value: Any,
    else_value: Any
) -> Any:
    """ if else branching """
    return if_value if condition else else_value

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

def rapply(
    functions: Container,
    value: Any
) -> Container:
    """ apply each function in container to value """
    return type(functions)(function(value) for function in functions)

def mostcolor(
    element: Element
) -> Integer:
    """ most common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return max(set(values), key=values.count)

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_2bee17df(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = trim(I)
    x1 = mostcolor(x0)
    x2 = repeat(x1, ONE)
    x3 = lbind(repeat, THREE)
    x4 = compose(x3, size)
    x5 = matcher(dedupe, x2)
    x6 = rbind(branch, identity)
    x7 = rbind(x6, x4)
    x8 = compose(x7, x5)
    x9 = compose(initset, x8)
    x10 = fork(rapply, x9, identity)
    x11 = compose(first, x10)
    x12 = apply(x11, x0)
    x13 = dmirror(x0)
    x14 = apply(x11, x13)
    x15 = dmirror(x14)
    x16 = ofcolor(x12, THREE)
    x17 = ofcolor(x15, THREE)
    x18 = combine(x16, x17)
    x19 = shift(x18, UNITY)
    x20 = fill(I, THREE, x19)
    return x20


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_2bee17df(inp)
        assert pred == _to_grid(expected), f"{name} failed"

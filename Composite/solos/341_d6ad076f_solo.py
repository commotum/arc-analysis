# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "d6ad076f"
SERIAL = "341"
URL    = "https://arcprize.org/play?task=d6ad076f"

# --- Code Golf Concepts ---
CONCEPTS = [
    "bridges",
    "connect_the_dots",
    "draw_line_from_point",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 2, 2, 2, 0, 0, 0, 0, 0],
    [0, 2, 2, 2, 2, 0, 0, 0, 0, 0],
    [0, 2, 2, 2, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [7, 7, 7, 7, 7, 7, 0, 0, 0, 0],
    [7, 7, 7, 7, 7, 7, 0, 0, 0, 0],
    [7, 7, 7, 7, 7, 7, 0, 0, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 2, 2, 2, 0, 0, 0, 0, 0],
    [0, 2, 2, 2, 2, 0, 0, 0, 0, 0],
    [0, 2, 2, 2, 2, 0, 0, 0, 0, 0],
    [0, 0, 8, 8, 0, 0, 0, 0, 0, 0],
    [0, 0, 8, 8, 0, 0, 0, 0, 0, 0],
    [0, 0, 8, 8, 0, 0, 0, 0, 0, 0],
    [7, 7, 7, 7, 7, 7, 0, 0, 0, 0],
    [7, 7, 7, 7, 7, 7, 0, 0, 0, 0],
    [7, 7, 7, 7, 7, 7, 0, 0, 0, 0],
], dtype=int)

E2_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 4, 4, 4, 0, 0, 0, 0, 0, 0],
    [0, 4, 4, 4, 0, 0, 0, 0, 0, 0],
    [0, 4, 4, 4, 0, 0, 0, 6, 6, 6],
    [0, 4, 4, 4, 0, 0, 0, 6, 6, 6],
    [0, 4, 4, 4, 0, 0, 0, 6, 6, 6],
    [0, 4, 4, 4, 0, 0, 0, 6, 6, 6],
    [0, 4, 4, 4, 0, 0, 0, 6, 6, 6],
    [0, 4, 4, 4, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 4, 4, 4, 0, 0, 0, 0, 0, 0],
    [0, 4, 4, 4, 0, 0, 0, 0, 0, 0],
    [0, 4, 4, 4, 0, 0, 0, 6, 6, 6],
    [0, 4, 4, 4, 8, 8, 8, 6, 6, 6],
    [0, 4, 4, 4, 8, 8, 8, 6, 6, 6],
    [0, 4, 4, 4, 8, 8, 8, 6, 6, 6],
    [0, 4, 4, 4, 0, 0, 0, 6, 6, 6],
    [0, 4, 4, 4, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E3_IN = np.array([
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 0],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 0],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 9, 9, 9, 9, 9, 9, 0],
    [0, 0, 0, 9, 9, 9, 9, 9, 9, 0],
], dtype=int)

E3_OUT = np.array([
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 0],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 0],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 0],
    [0, 0, 0, 0, 8, 8, 8, 8, 0, 0],
    [0, 0, 0, 0, 8, 8, 8, 8, 0, 0],
    [0, 0, 0, 0, 8, 8, 8, 8, 0, 0],
    [0, 0, 0, 0, 8, 8, 8, 8, 0, 0],
    [0, 0, 0, 0, 8, 8, 8, 8, 0, 0],
    [0, 0, 0, 9, 9, 9, 9, 9, 9, 0],
    [0, 0, 0, 9, 9, 9, 9, 9, 9, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 2, 2, 2],
    [1, 1, 1, 1, 0, 0, 0, 2, 2, 2],
    [1, 1, 1, 1, 0, 0, 0, 2, 2, 2],
    [1, 1, 1, 1, 0, 0, 0, 2, 2, 2],
    [1, 1, 1, 1, 0, 0, 0, 2, 2, 2],
    [1, 1, 1, 1, 0, 0, 0, 2, 2, 2],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

T_OUT = np.array([
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 2, 2, 2],
    [1, 1, 1, 1, 8, 8, 8, 2, 2, 2],
    [1, 1, 1, 1, 8, 8, 8, 2, 2, 2],
    [1, 1, 1, 1, 8, 8, 8, 2, 2, 2],
    [1, 1, 1, 1, 8, 8, 8, 2, 2, 2],
    [1, 1, 1, 1, 0, 0, 0, 2, 2, 2],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
R=range
L=len
def p(g):
 for i in R(4):
  g=list(map(list,zip(*g[::-1])))
  h,w,I=L(g),L(g[0]),0
  for r in R(h):
   if len(set(g[r]))>2 and 8 not in g[r]:
    S=C=0
    if I>0:
     for c in R(w):
      if g[r][c]>0 and C==0 and not S:S=1;C=g[r][c]
      if g[r][c]>0 and g[r][c]!=C:S=0
      if S==1 and g[r][c]==0:g[r][c]=8
    I+=1
   elif I>0:
    for c in R(w):
     if g[r-1][c]==8:g[r-1][c]=0
    I=0
 return g


# --- Code Golf Solution (Compressed) ---
def q(g):
    return [(g := [[g[i][j] or (9 > j >= 2 < len({*min(g[i - 1:][:3])})) * 8 for i in R] for j in R]) for R in [range(10)] * 2][1]


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, randint, sample, uniform

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

Piece = Union[Grid, Patch]

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

def asobject(
    grid: Grid
) -> Object:
    """ conversion of grid to object """
    return frozenset((v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r))

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

def generate_d6ad076f(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(8, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    inh = unifint(diff_lb, diff_ub, (3, h))
    inw = unifint(diff_lb, diff_ub, (3, w))
    bgc, c1, c2 = sample(cols, 3)
    itv = interval(0, inh, 1)
    loci2i = unifint(diff_lb, diff_ub, (2, inh - 1))
    loci2 = itv[loci2i]
    itv = itv[:loci2i-1][::-1]
    loci1i = unifint(diff_lb, diff_ub, (0, len(itv) - 1))
    loci1 = itv[loci1i]
    cp = randint(1, inw - 2)
    ajs = randint(0, cp - 1)
    aje = randint(cp + 1, inw - 1)
    bjs = randint(0, cp - 1)
    bje = randint(cp + 1, inw - 1)
    obja = backdrop(frozenset({(0, ajs), (loci1, aje)}))
    objb = backdrop(frozenset({(loci2, bjs), (inh - 1, bje)}))
    c = canvas(bgc, (inh, inw))
    c = fill(c, c1, obja)
    c = fill(c, c2, objb)
    obj = asobject(c)
    loci = randint(0, h - inh)
    locj = randint(0, w - inw)
    loc = (loci, locj)
    obj = shift(obj, loc)
    gi = canvas(bgc, (h, w))
    gi = paint(gi, obj)
    midobj = backdrop(frozenset({(loci1 + 1, max(ajs, bjs) + 1), (loci2 - 1, min(aje, bje) - 1)}))
    go = fill(gi, 8, shift(midobj, loc))
    if choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Numerical = Union[Integer, IntegerTuple]

IntegerSet = FrozenSet[Integer]

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

ContainerContainer = Container[Container]

EIGHT = 8

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

def contained(
    value: Any,
    container: Container
) -> Boolean:
    """ element of """
    return value in container

def combine(
    a: Container,
    b: Container
) -> Container:
    """ union """
    return type(a)((*a, *b))

def merge(
    containers: ContainerContainer
) -> Container:
    """ merging """
    return type(containers)(e for c in containers for e in c)

def maximum(
    container: IntegerSet
) -> Integer:
    """ maximum """
    return max(container, default=0)

def minimum(
    container: IntegerSet
) -> Integer:
    """ minimum """
    return min(container, default=0)

def initset(
    value: Any
) -> FrozenSet:
    """ initialize container """
    return frozenset({value})

def increment(
    x: Numerical
) -> Numerical:
    """ incrementing """
    return x + 1 if isinstance(x, int) else (x[0] + 1, x[1] + 1)

def decrement(
    x: Numerical
) -> Numerical:
    """ decrementing """
    return x - 1 if isinstance(x, int) else (x[0] - 1, x[1] - 1)

def sfilter(
    container: Container,
    condition: Callable
) -> Container:
    """ keep elements in container that satisfy condition """
    return type(container)(e for e in container if condition(e))

def extract(
    container: Container,
    condition: Callable
) -> Any:
    """ first element of container that satisfies condition """
    return next(e for e in container if condition(e))

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

def insert(
    value: Any,
    container: FrozenSet
) -> FrozenSet:
    """ insert item into container """
    return container.union(frozenset({value}))

def astuple(
    a: Integer,
    b: Integer
) -> IntegerTuple:
    """ constructs a tuple """
    return (a, b)

def product(
    a: Container,
    b: Container
) -> FrozenSet:
    """ cartesian product """
    return frozenset((i, j) for j in b for i in a)

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

def partition(
    grid: Grid
) -> Objects:
    """ each cell with the same value part of the same object """
    return frozenset(
        frozenset(
            (v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value
        ) for value in palette(grid)
    )

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

def hmatching(
    a: Patch,
    b: Patch
) -> Boolean:
    """ whether there exists a row for which both patches have cells """
    return len(set(i for i, j in toindices(a)) & set(i for i, j in toindices(b))) > 0

def manhattan(
    a: Patch,
    b: Patch
) -> Integer:
    """ closest manhattan distance between two patches """
    return min(abs(ai - bi) + abs(aj - bj) for ai, aj in toindices(a) for bi, bj in toindices(b))

def adjacent(
    a: Patch,
    b: Patch
) -> Boolean:
    """ whether two patches are adjacent """
    return manhattan(a, b) == 1

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

def verify_d6ad076f(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = partition(I)
    x1 = product(x0, x0)
    x2 = fork(equality, first, last)
    x3 = compose(flip, x2)
    x4 = sfilter(x1, x3)
    x5 = fork(adjacent, first, last)
    x6 = compose(flip, x5)
    x7 = extract(x4, x6)
    x8 = totuple(x7)
    x9 = first(x8)
    x10 = last(x8)
    x11 = combine(x9, x10)
    x12 = leftmost(x11)
    x13 = increment(x12)
    x14 = rightmost(x11)
    x15 = decrement(x14)
    x16 = apply(uppermost, x8)
    x17 = maximum(x16)
    x18 = increment(x17)
    x19 = apply(lowermost, x8)
    x20 = minimum(x19)
    x21 = decrement(x20)
    x22 = apply(leftmost, x8)
    x23 = maximum(x22)
    x24 = increment(x23)
    x25 = apply(rightmost, x8)
    x26 = minimum(x25)
    x27 = decrement(x26)
    x28 = uppermost(x11)
    x29 = increment(x28)
    x30 = lowermost(x11)
    x31 = decrement(x30)
    x32 = hmatching(x9, x10)
    x33 = branch(x32, x13, x24)
    x34 = branch(x32, x15, x27)
    x35 = branch(x32, x21, x31)
    x36 = branch(x32, x18, x29)
    x37 = astuple(x35, x34)
    x38 = astuple(x36, x33)
    x39 = initset(x38)
    x40 = insert(x37, x39)
    x41 = backdrop(x40)
    x42 = merge(x7)
    x43 = toindices(x42)
    x44 = rbind(contained, x43)
    x45 = compose(flip, x44)
    x46 = sfilter(x41, x45)
    x47 = fill(I, EIGHT, x46)
    return x47


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_d6ad076f(inp)
        assert pred == _to_grid(expected), f"{name} failed"

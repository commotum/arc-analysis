# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "bbc9ae5d"
SERIAL = "295"
URL    = "https://arcprize.org/play?task=bbc9ae5d"

# --- Code Golf Concepts ---
CONCEPTS = [
    "pattern_expansion",
    "image_expansion",
]

# --- Example Grids ---
E1_IN = np.array([
    [1, 1, 0, 0, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [1, 1, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [1, 1, 1, 1, 0, 0],
], dtype=int)

E2_IN = np.array([
    [2, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [2, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 0, 0, 0, 0, 0, 0],
    [2, 2, 2, 0, 0, 0, 0, 0],
    [2, 2, 2, 2, 0, 0, 0, 0],
], dtype=int)

E3_IN = np.array([
    [5, 5, 5, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [5, 5, 5, 0, 0, 0, 0, 0, 0, 0],
    [5, 5, 5, 5, 0, 0, 0, 0, 0, 0],
    [5, 5, 5, 5, 5, 0, 0, 0, 0, 0],
    [5, 5, 5, 5, 5, 5, 0, 0, 0, 0],
    [5, 5, 5, 5, 5, 5, 5, 0, 0, 0],
], dtype=int)

E4_IN = np.array([
    [8, 8, 8, 8, 0, 0],
], dtype=int)

E4_OUT = np.array([
    [8, 8, 8, 8, 0, 0],
    [8, 8, 8, 8, 8, 0],
    [8, 8, 8, 8, 8, 8],
], dtype=int)

E5_IN = np.array([
    [7, 0, 0, 0, 0, 0],
], dtype=int)

E5_OUT = np.array([
    [7, 0, 0, 0, 0, 0],
    [7, 7, 0, 0, 0, 0],
    [7, 7, 7, 0, 0, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

T_OUT = np.array([
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(g,L=len,R=range):
 g=g[0]
 C=g[0]
 T=L([x for x in g if x>0])
 w=R(L(g))
 h=R(L(g)//2)
 X=[[0 for x in w] for y in h]
 for r in h:
  for c in w:
   if c<T:X[r][c]=C
  T+=1
 return X


# --- Code Golf Solution (Compressed) ---
def q(g):
    return [(P := g[0])] + [(P := (P[:1] + P[:-1])) for x in P[2::2]]


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

ContainerContainer = Container[Container]

def merge(
    containers: ContainerContainer
) -> Container:
    """ merging """
    return type(containers)(e for c in containers for e in c)

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

def generate_bbc9ae5d(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    w = unifint(diff_lb, diff_ub, (2, 15))
    w = w * 2
    locinv = unifint(diff_lb, diff_ub, (2, w))
    locj = w - locinv
    loc = (0, locj)
    c1 = choice(cols)
    remcols = remove(c1, cols)
    ln1 = connect((0, 0), (0, locj))
    remobj = connect((0, locj+1), (0, w - 1))
    numc = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(remcols, numc)
    remobj = {(choice(ccols), ij) for ij in remobj}
    gi = canvas(-1, (1, w))
    go = canvas(-1, (w//2, w))
    ln2 = shoot(loc, (1, 1))
    gi = fill(gi, c1, ln1)
    gi = paint(gi, remobj)
    go = fill(go, c1, mapply(rbind(shoot, (0, -1)), ln2))
    for c, ij in remobj:
        go = fill(go, c, shoot(ij, (1, 1)))
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Numerical = Union[Integer, IntegerTuple]

Piece = Union[Grid, Patch]

ORIGIN = (0, 0)

UNITY = (1, 1)

def halve(
    n: Numerical
) -> Numerical:
    """ scaling by one half """
    return n // 2 if isinstance(n, int) else (n[0] // 2, n[1] // 2)

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

def astuple(
    a: Integer,
    b: Integer
) -> IntegerTuple:
    """ constructs a tuple """
    return (a, b)

def compose(
    outer: Callable,
    inner: Callable
) -> Callable:
    """ function composition """
    return lambda x: outer(inner(x))

def fork(
    outer: Callable,
    a: Callable,
    b: Callable
) -> Callable:
    """ creates a wrapper function """
    return lambda x: outer(a(x), b(x))

def width(
    piece: Piece
) -> Integer:
    """ width of grid or patch """
    if len(piece) == 0:
        return 0
    if isinstance(piece, tuple):
        return len(piece[0])
    return rightmost(piece) - leftmost(piece) + 1

def recolor(
    value: Integer,
    patch: Patch
) -> Object:
    """ recolor patch """
    return frozenset((value, index) for index in toindices(patch))

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

def asobject(
    grid: Grid
) -> Object:
    """ conversion of grid to object """
    return frozenset((v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r))

def index(
    grid: Grid,
    loc: IntegerTuple
) -> Integer:
    """ color at location """
    i, j = loc
    h, w = len(grid), len(grid[0])
    if not (0 <= i < h and 0 <= j < w):
        return None
    return grid[loc[0]][loc[1]]

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_bbc9ae5d(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = index(I, ORIGIN)
    x1 = width(I)
    x2 = halve(x1)
    x3 = astuple(x2, x1)
    x4 = canvas(x0, x3)
    x5 = rbind(shoot, UNITY)
    x6 = compose(x5, last)
    x7 = fork(recolor, first, x6)
    x8 = asobject(I)
    x9 = mapply(x7, x8)
    x10 = paint(x4, x9)
    return x10


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("E4", E4_IN, E4_OUT),
        ("E5", E5_IN, E5_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_bbc9ae5d(inp)
        assert pred == _to_grid(expected), f"{name} failed"

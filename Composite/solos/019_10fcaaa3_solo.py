# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "10fcaaa3"
SERIAL = "019"
URL    = "https://arcprize.org/play?task=10fcaaa3"

# --- Code Golf Concepts ---
CONCEPTS = [
    "pattern_expansion",
    "image_repetition",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 0, 0],
    [0, 5, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [8, 0, 8, 0, 8, 0, 8, 0],
    [0, 5, 0, 0, 0, 5, 0, 0],
    [8, 0, 8, 0, 8, 0, 8, 0],
    [0, 5, 0, 0, 0, 5, 0, 0],
], dtype=int)

E2_IN = np.array([
    [0, 0, 6, 0],
    [0, 0, 0, 0],
    [0, 6, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [0, 0, 6, 0, 0, 0, 6, 0],
    [8, 8, 8, 8, 8, 8, 8, 8],
    [0, 6, 0, 8, 0, 6, 0, 8],
    [8, 0, 6, 0, 8, 0, 6, 0],
    [8, 8, 8, 8, 8, 8, 8, 8],
    [0, 6, 0, 0, 0, 6, 0, 0],
], dtype=int)

E3_IN = np.array([
    [0, 0, 0],
    [0, 4, 0],
    [0, 0, 0],
    [0, 0, 0],
    [4, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [8, 0, 8, 8, 0, 8],
    [0, 4, 0, 0, 4, 0],
    [8, 0, 8, 8, 0, 8],
    [0, 8, 8, 0, 8, 0],
    [4, 0, 0, 4, 0, 0],
    [8, 8, 8, 8, 8, 8],
    [0, 4, 0, 0, 4, 0],
    [8, 0, 8, 8, 0, 8],
    [0, 8, 8, 0, 8, 0],
    [4, 0, 0, 4, 0, 0],
], dtype=int)

E4_IN = np.array([
    [0, 0, 0, 0],
    [0, 2, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
], dtype=int)

E4_OUT = np.array([
    [8, 0, 8, 0, 8, 0, 8, 0],
    [0, 2, 0, 0, 0, 2, 0, 0],
    [8, 0, 8, 0, 8, 0, 8, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [8, 0, 8, 0, 8, 0, 8, 0],
    [0, 2, 0, 0, 0, 2, 0, 0],
    [8, 0, 8, 0, 8, 0, 8, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 3, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 3, 0],
    [0, 0, 0, 0, 0],
    [0, 3, 0, 0, 0],
], dtype=int)

T_OUT = np.array([
    [0, 3, 0, 0, 0, 0, 3, 0, 0, 0],
    [8, 0, 8, 0, 0, 8, 0, 8, 0, 0],
    [0, 0, 8, 0, 8, 0, 0, 8, 0, 8],
    [0, 0, 0, 3, 0, 0, 0, 0, 3, 0],
    [8, 0, 8, 0, 8, 8, 0, 8, 0, 8],
    [8, 3, 8, 0, 0, 8, 3, 8, 0, 0],
    [8, 3, 8, 0, 0, 8, 3, 8, 0, 0],
    [8, 0, 8, 0, 0, 8, 0, 8, 0, 0],
    [0, 0, 8, 0, 8, 0, 0, 8, 0, 8],
    [0, 0, 0, 3, 0, 0, 0, 0, 3, 0],
    [8, 0, 8, 0, 8, 8, 0, 8, 0, 8],
    [0, 3, 0, 0, 0, 0, 3, 0, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
L=len
R=range
def p(g):
 g=[r[:]+r[:]for r in g]+[r[:]+r[:]for r in g]
 h,w=L(g),L(g[0])
 for r in R(h):
  for c in R(w):
   C=g[r][c]
   if C>0 and C!=8:
    for i,j in[[1,1],[-1,-1],[-1,1],[1,-1]]:
     if i+r>=0 and j+c>=0 and i+r<h and j+c<w:
      if g[i+r][j+c]==0:g[i+r][j+c]=8
 return g


# --- Code Golf Solution (Compressed) ---
def q(g, n=7):
    return -n * g or p(-~(n > 5) * [(g := [r.pop() or (x * -1 or 0) % -8 & 8 for x in [0] + g[:-1]]) for *r, in zip(*g)], n - 1)


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

def ineighbors(
    loc: IntegerTuple
) -> Indices:
    """ diagonally adjacent indices """
    return frozenset({(loc[0] - 1, loc[1] - 1), (loc[0] - 1, loc[1] + 1), (loc[0] + 1, loc[1] - 1), (loc[0] + 1, loc[1] + 1)})

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

def hconcat(
    a: Grid,
    b: Grid
) -> Grid:
    """ concatenate two grids horizontally """
    return tuple(i + j for i, j in zip(a, b))

def vconcat(
    a: Grid,
    b: Grid
) -> Grid:
    """ concatenate two grids vertically """
    return a + b

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

def generate_10fcaaa3(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(8, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 15))
    w = unifint(diff_lb, diff_ub, (2, 15))
    ncells = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 6)))
    ncols = unifint(diff_lb, diff_ub, (1, 8))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    ccols = sample(remcols, ncols)
    c = canvas(bgc, (h, w))
    inds = asindices(c)
    locs = frozenset(sample(totuple(inds), ncells))
    obj = frozenset({(choice(ccols), ij) for ij in locs})
    gi = paint(c, obj)
    go = hconcat(gi, gi)
    go = vconcat(go, go)
    fullocs = locs | shift(locs, (0, w)) | shift(locs, (h, 0)) | shift(locs, (h, w))
    nbhs = mapply(ineighbors, fullocs)
    topaint = nbhs & ofcolor(go, bgc)
    go = fill(go, 8, topaint)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Element = Union[Object, Grid]

EIGHT = 8

def difference(
    a: Container,
    b: Container
) -> Container:
    """ difference """
    return type(a)(e for e in a if e not in b)

def mostcolor(
    element: Element
) -> Integer:
    """ most common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return max(set(values), key=values.count)

def underfill(
    grid: Grid,
    value: Integer,
    patch: Patch
) -> Grid:
    """ fill value at indices that are background """
    h, w = len(grid), len(grid[0])
    bg = mostcolor(grid)
    grid_filled = list(list(row) for row in grid)
    for i, j in toindices(patch):
        if 0 <= i < h and 0 <= j < w:
            if grid_filled[i][j] == bg:
                grid_filled[i][j] = value
    return tuple(tuple(row) for row in grid_filled)

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_10fcaaa3(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = hconcat(I, I)
    x1 = vconcat(x0, x0)
    x2 = asindices(x1)
    x3 = mostcolor(I)
    x4 = ofcolor(x1, x3)
    x5 = difference(x2, x4)
    x6 = mapply(ineighbors, x5)
    x7 = underfill(x1, EIGHT, x6)
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
        pred = verify_10fcaaa3(inp)
        assert pred == _to_grid(expected), f"{name} failed"

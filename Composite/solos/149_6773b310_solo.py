# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "6773b310"
SERIAL = "149"
URL    = "https://arcprize.org/play?task=6773b310"

# --- Code Golf Concepts ---
CONCEPTS = [
    "detect_grid",
    "separate_images",
    "count_tiles",
    "associate_colors_to_numbers",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0],
    [6, 0, 0, 8, 0, 6, 0, 8, 0, 0, 6],
    [0, 0, 6, 8, 0, 0, 0, 8, 0, 6, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 6, 0, 8, 0, 0, 6, 8, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0],
    [0, 6, 0, 8, 0, 0, 0, 8, 6, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 6, 8, 0, 0, 0, 8, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 6, 0],
    [0, 0, 0, 8, 6, 0, 0, 8, 0, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [1, 0, 1],
    [1, 0, 0],
    [0, 0, 0],
], dtype=int)

E2_IN = np.array([
    [6, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 6, 8, 0, 0, 6],
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [6, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 6, 0],
    [0, 0, 0, 8, 0, 0, 6, 8, 6, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0],
    [6, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0],
    [0, 6, 0, 8, 0, 6, 0, 8, 0, 0, 6],
], dtype=int)

E2_OUT = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [1, 0, 0],
], dtype=int)

E3_IN = np.array([
    [0, 0, 0, 8, 0, 6, 0, 8, 0, 0, 6],
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 6, 0],
    [0, 6, 0, 8, 0, 6, 0, 8, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0],
    [0, 0, 0, 8, 0, 6, 0, 8, 0, 0, 0],
    [6, 0, 0, 8, 0, 0, 0, 8, 0, 6, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 8, 0, 0, 0, 8, 6, 0, 0],
    [0, 6, 0, 8, 0, 0, 0, 8, 0, 0, 6],
    [0, 0, 0, 8, 6, 0, 0, 8, 0, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [0, 1, 1],
    [0, 0, 0],
    [0, 0, 1],
], dtype=int)

E4_IN = np.array([
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 6],
    [0, 0, 6, 8, 0, 0, 0, 8, 6, 0, 0],
    [0, 0, 0, 8, 0, 6, 0, 8, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 8, 0, 6, 0, 8, 0, 0, 0],
    [6, 0, 0, 8, 0, 0, 6, 8, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 6, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0],
    [0, 0, 6, 8, 0, 0, 0, 8, 6, 0, 0],
    [0, 0, 0, 8, 0, 6, 0, 8, 0, 0, 0],
], dtype=int)

E4_OUT = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [0, 0, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 0, 8, 0, 0, 0, 8, 6, 0, 6],
    [0, 6, 0, 8, 0, 0, 6, 8, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 6, 0],
    [0, 0, 6, 8, 0, 6, 0, 8, 0, 0, 0],
    [0, 0, 0, 8, 6, 0, 0, 8, 0, 0, 6],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 6, 8, 0, 0, 0, 8, 0, 0, 0],
    [6, 0, 0, 8, 0, 0, 0, 8, 0, 6, 0],
    [0, 0, 0, 8, 0, 6, 0, 8, 0, 0, 0],
], dtype=int)

T_OUT = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
 A=range
 c=[[0]*3for _ in A(3)]
 for E in A(3):
  for k in A(3):
   W=0
   for l in A(3):
    for J in A(3):
     if j[E*4+l][k*4+J]==6:W+=1
   c[E][k]=1if W>=2else 0
 return c


# --- Code Golf Solution (Compressed) ---
def q(g):
    return g[3:] and [p([*zip(*g[i:i + 3])]) for i in [0, 4, 8]] or sum(b'%r/' % g) % 5


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

def generate_6773b310(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(1, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 5))
    w = unifint(diff_lb, diff_ub, (2, 5))
    nh = unifint(diff_lb, diff_ub, (2, 5))
    nw = unifint(diff_lb, diff_ub, (2, 5))
    bgc, linc, fgc = sample(cols, 3)
    fullh = h * nh + (nh - 1)
    fullw = w * nw + (nw - 1)
    c = canvas(linc, (fullh, fullw))
    smallc = canvas(bgc, (h, w))
    llocs = set()
    for a in range(0, fullh, h + 1):
        for b in range(0, fullw, w + 1):
            llocs.add((a, b))
    llocs = tuple(llocs)
    nbldev = unifint(diff_lb, diff_ub, (0, (nh * nw) // 2))
    nbl = choice((nbldev, nh * nw - nbldev))
    nbl = min(max(1, nbl), nh * nw - 1)
    bluelocs = sample(llocs, nbl)
    bglocs = difference(llocs, bluelocs)
    inds = totuple(asindices(smallc))
    gi = tuple(e for e in c)
    go = canvas(bgc, (nh, nw))
    for ij in bluelocs:
        subg = asobject(fill(smallc, fgc, sample(inds, 2)))
        gi = paint(gi, shift(subg, ij))
        a, b = ij
        loci = a // (h+1)
        locj = b // (w+1)
        go = fill(go, 1, {(loci, locj)})
    for ij in bglocs:
        subg = asobject(fill(smallc, fgc, sample(inds, 1)))
        gi = paint(gi, shift(subg, ij))
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Numerical = Union[Integer, IntegerTuple]

IntegerSet = FrozenSet[Integer]

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

ContainerContainer = Container[Container]

ONE = 1

F = False

T = True

def divide(
    a: Numerical,
    b: Numerical
) -> Numerical:
    """ floor division """
    if isinstance(a, int) and isinstance(b, int):
        return a // b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] // b[0], a[1] // b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a // b[0], a // b[1])
    return (a[0] // b, a[1] // b)

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

def valmax(
    container: Container,
    compfunc: Callable
) -> Integer:
    """ maximum by custom function """
    return compfunc(max(container, key=compfunc, default=0))

def argmin(
    container: Container,
    compfunc: Callable
) -> Any:
    """ smallest item by custom order """
    return min(container, key=compfunc, default=None)

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

def astuple(
    a: Integer,
    b: Integer
) -> IntegerTuple:
    """ constructs a tuple """
    return (a, b)

def matcher(
    function: Callable,
    target: Any
) -> Callable:
    """ construction of equality function """
    return lambda x: function(x) == target

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

def mostcolor(
    element: Element
) -> Integer:
    """ most common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return max(set(values), key=values.count)

def height(
    piece: Piece
) -> Integer:
    """ height of grid or patch """
    if len(piece) == 0:
        return 0
    if isinstance(piece, tuple):
        return len(piece)
    return lowermost(piece) - uppermost(piece) + 1

def width(
    piece: Piece
) -> Integer:
    """ width of grid or patch """
    if len(piece) == 0:
        return 0
    if isinstance(piece, tuple):
        return len(piece[0])
    return rightmost(piece) - leftmost(piece) + 1

def shape(
    piece: Piece
) -> IntegerTuple:
    """ height and width of grid or patch """
    return (height(piece), width(piece))

def colorcount(
    element: Element,
    value: Integer
) -> Integer:
    """ number of cells with color """
    if isinstance(element, tuple):
        return sum(row.count(value) for row in element)
    return sum(v == value for v, _ in element)

def ulcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of upper left corner """
    return tuple(map(min, zip(*toindices(patch))))

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

def vline(
    patch: Patch
) -> Boolean:
    """ whether the piece forms a vertical line """
    return height(patch) == len(patch) and width(patch) == 1

def hline(
    patch: Patch
) -> Boolean:
    """ whether the piece forms a horizontal line """
    return width(patch) == len(patch) and height(patch) == 1

def palette(
    element: Element
) -> IntegerSet:
    """ colors occurring in object or grid """
    if isinstance(element, tuple):
        return frozenset({v for r in element for v in r})
    return frozenset({v for v, _ in element})

def color(
    obj: Object
) -> Integer:
    """ color of object """
    return next(iter(obj))[0]

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

def hconcat(
    a: Grid,
    b: Grid
) -> Grid:
    """ concatenate two grids horizontally """
    return tuple(i + j for i, j in zip(a, b))

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

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_6773b310(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = frontiers(I)
    x1 = merge(x0)
    x2 = color(x1)
    x3 = shape(I)
    x4 = canvas(x2, x3)
    x5 = hconcat(I, x4)
    x6 = palette(I)
    x7 = remove(x2, x6)
    x8 = lbind(colorcount, I)
    x9 = argmin(x7, x8)
    x10 = other(x7, x9)
    x11 = objects(x5, F, T, T)
    x12 = rbind(colorcount, x9)
    x13 = valmax(x11, x12)
    x14 = rbind(colorcount, x9)
    x15 = matcher(x14, x13)
    x16 = sfilter(x11, x15)
    x17 = apply(ulcorner, x16)
    x18 = first(x11)
    x19 = shape(x18)
    x20 = increment(x19)
    x21 = rbind(divide, x20)
    x22 = apply(x21, x17)
    x23 = sfilter(x0, hline)
    x24 = size(x23)
    x25 = sfilter(x0, vline)
    x26 = size(x25)
    x27 = astuple(x24, x26)
    x28 = increment(x27)
    x29 = canvas(x10, x28)
    x30 = fill(x29, ONE, x22)
    return x30


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("E4", E4_IN, E4_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_6773b310(inp)
        assert pred == _to_grid(expected), f"{name} failed"

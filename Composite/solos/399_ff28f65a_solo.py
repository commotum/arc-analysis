# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "ff28f65a"
SERIAL = "399"
URL    = "https://arcprize.org/play?task=ff28f65a"

# --- Code Golf Concepts ---
CONCEPTS = [
    "count_shapes",
    "associate_images_to_numbers",
]

# --- Example Grids ---
E1_IN = np.array([
    [2, 2, 0, 0, 0],
    [2, 2, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [1, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
], dtype=int)

E2_IN = np.array([
    [0, 0, 0, 0, 0],
    [0, 2, 2, 0, 0],
    [0, 2, 2, 0, 0],
    [0, 0, 0, 2, 2],
    [0, 0, 0, 2, 2],
], dtype=int)

E2_OUT = np.array([
    [1, 0, 1],
    [0, 0, 0],
    [0, 0, 0],
], dtype=int)

E3_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 2, 2, 0, 0, 0, 0],
    [0, 2, 2, 0, 2, 2, 0],
    [0, 0, 0, 0, 2, 2, 0],
    [0, 0, 2, 2, 0, 0, 0],
    [0, 0, 2, 2, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [0, 0, 0],
], dtype=int)

E4_IN = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 2, 2, 0, 0, 0],
    [0, 2, 2, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 2, 2, 0, 0],
    [0, 0, 2, 2, 0, 0],
], dtype=int)

E4_OUT = np.array([
    [1, 0, 1],
    [0, 0, 0],
    [0, 0, 0],
], dtype=int)

E5_IN = np.array([
    [0, 0, 0],
    [0, 2, 2],
    [0, 2, 2],
], dtype=int)

E5_OUT = np.array([
    [1, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
], dtype=int)

E6_IN = np.array([
    [0, 0, 0, 0, 2, 2, 0],
    [0, 0, 0, 0, 2, 2, 0],
    [0, 2, 2, 0, 0, 0, 0],
    [0, 2, 2, 0, 2, 2, 0],
    [0, 0, 0, 0, 2, 2, 0],
    [0, 2, 2, 0, 0, 0, 0],
    [0, 2, 2, 0, 0, 0, 0],
], dtype=int)

E6_OUT = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
], dtype=int)

E7_IN = np.array([
    [0, 0, 0, 0, 2, 2, 0],
    [0, 2, 2, 0, 2, 2, 0],
    [0, 2, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 2, 2],
    [2, 2, 0, 0, 0, 2, 2],
    [2, 2, 0, 2, 2, 0, 0],
    [0, 0, 0, 2, 2, 0, 0],
], dtype=int)

E7_OUT = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1],
], dtype=int)

E8_IN = np.array([
    [0, 0, 2, 2, 0, 2, 2],
    [0, 0, 2, 2, 0, 2, 2],
    [2, 2, 0, 0, 0, 0, 0],
    [2, 2, 0, 2, 2, 0, 0],
    [0, 0, 0, 2, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E8_OUT = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 0, 2, 2, 0],
    [2, 2, 0, 2, 2, 0],
    [2, 2, 0, 0, 0, 0],
    [0, 0, 2, 2, 0, 0],
    [0, 0, 2, 2, 0, 0],
    [0, 0, 0, 0, 0, 0],
], dtype=int)

T_OUT = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [0, 0, 0],
], dtype=int)

T2_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [2, 2, 0, 2, 2, 0, 0],
    [2, 2, 0, 2, 2, 0, 0],
    [0, 0, 0, 0, 0, 2, 2],
    [0, 0, 2, 2, 0, 2, 2],
    [0, 0, 2, 2, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
], dtype=int)

T2_OUT = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
], dtype=int)

T3_IN = np.array([
    [2, 2, 0, 2, 2, 0, 0],
    [2, 2, 0, 2, 2, 0, 0],
    [0, 0, 0, 0, 0, 2, 2],
    [0, 2, 2, 0, 0, 2, 2],
    [0, 2, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 2, 0],
    [0, 0, 0, 0, 2, 2, 0],
], dtype=int)

T3_OUT = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j,A=0):
 c={1:[[1,0,0],[0,0,0],[0,0,0]],2:[[1,0,1],[0,0,0],[0,0,0]],3:[[1,0,1],[0,1,0],[0,0,0]],4:[[1,0,1],[0,1,0],[1,0,0]],5:[[1,0,1],[0,1,0],[1,0,1]]}
 for E in range(0,len(j[0])-2+1,1):
  for k in range(0,len(j)-2+1,1):
   W=j[E:E+2];W=[R[k:k+2]for R in W];l=[i for s in W for i in s]
   if min(l)>0:A+=1
 return c[A]


# --- Code Golf Solution (Compressed) ---
def q(g):
    return [*zip(*[iter(sum(sum(g, [])) % 7 * [1, 0] + [0] * 9)] * 3)][:3]


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

ContainerContainer = Container[Container]

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

def generate_ff28f65a(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2))
    mpr = {1: (0, 0), 2: (0, 2), 3: (1, 1), 4: (2, 0), 5: (2, 2)}
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc = choice(cols)
    nred = randint(1, 5)
    gi = canvas(bgc, (h, w))
    succ = 0
    tr = 0
    maxtr = 5 * nred
    inds = asindices(gi)
    while tr < maxtr and succ < nred:
        tr += 1
        oh = randint(1, h//2+1)
        ow = randint(1, w//2+1)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        loci, locj = loc
        bd = backdrop(frozenset({(loci, locj), (loci+oh-1, locj+ow-1)}))
        if bd.issubset(inds):
            succ += 1
            inds = (inds - bd) - mapply(dneighbors, bd)
            gi = fill(gi, 2, bd)
    nblue = succ
    namt = unifint(diff_lb, diff_ub, (0, nred * 2))
    succ = 0
    tr = 0
    maxtr = 5 * namt
    remcols = remove(bgc, cols)
    tr += 1
    while tr < maxtr and succ < namt:
        tr += 1
        oh = randint(1, h//2+1)
        ow = randint(1, w//2+1)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        loci, locj = loc
        bd = backdrop(frozenset({(loci, locj), (loci+oh-1, locj+ow-1)}))
        if bd.issubset(inds):
            succ += 1
            inds = (inds - bd) - mapply(dneighbors, bd)
            gi = fill(gi, choice(remcols), bd)
    go = canvas(bgc, (3, 3))
    for k in range(nblue):
        go = fill(go, 1, {mpr[k+1]})
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Numerical = Union[Integer, IntegerTuple]

IntegerSet = FrozenSet[Integer]

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

ZERO = 0

ONE = 1

TWO = 2

THREE = 3

NINE = 9

F = False

T = True

def double(
    n: Numerical
) -> Numerical:
    """ scaling by two """
    return n * 2 if isinstance(n, int) else (n[0] * 2, n[1] * 2)

def size(
    container: Container
) -> Integer:
    """ cardinality """
    return len(container)

def argmax(
    container: Container,
    compfunc: Callable
) -> Any:
    """ largest item by custom order """
    return max(container, key=compfunc, default=None)

def tojvec(
    j: Integer
) -> IntegerTuple:
    """ vector pointing horizontally """
    return (0, j)

def astuple(
    a: Integer,
    b: Integer
) -> IntegerTuple:
    """ constructs a tuple """
    return (a, b)

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

def colorfilter(
    objs: Objects,
    value: Integer
) -> Objects:
    """ filter objects by color """
    return frozenset(obj for obj in objs if next(iter(obj))[0] == value)

def crop(
    grid: Grid,
    start: IntegerTuple,
    dims: IntegerTuple
) -> Grid:
    """ subgrid specified by start and dimension """
    return tuple(r[start[1]:start[1]+dims[1]] for r in grid[start[0]:start[0]+dims[0]])

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

def palette(
    element: Element
) -> IntegerSet:
    """ colors occurring in object or grid """
    if isinstance(element, tuple):
        return frozenset({v for r in element for v in r})
    return frozenset({v for v, _ in element})

def hconcat(
    a: Grid,
    b: Grid
) -> Grid:
    """ concatenate two grids horizontally """
    return tuple(i + j for i, j in zip(a, b))

def hsplit(
    grid: Grid,
    n: Integer
) -> Tuple:
    """ split grid horizontally """
    h, w = len(grid), len(grid[0]) // n
    offset = len(grid[0]) % n != 0
    return tuple(crop(grid, (0, w * i + i * offset), (h, w)) for i in range(n))

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_ff28f65a(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = palette(I)
    x1 = remove(TWO, x0)
    x2 = lbind(colorcount, I)
    x3 = argmax(x1, x2)
    x4 = shape(I)
    x5 = canvas(x3, x4)
    x6 = hconcat(I, x5)
    x7 = objects(x6, T, F, T)
    x8 = colorfilter(x7, TWO)
    x9 = size(x8)
    x10 = double(x9)
    x11 = interval(ZERO, x10, TWO)
    x12 = apply(tojvec, x11)
    x13 = astuple(ONE, NINE)
    x14 = canvas(x3, x13)
    x15 = fill(x14, ONE, x12)
    x16 = hsplit(x15, THREE)
    x17 = merge(x16)
    return x17


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("E4", E4_IN, E4_OUT),
        ("E5", E5_IN, E5_OUT),
        ("E6", E6_IN, E6_OUT),
        ("E7", E7_IN, E7_OUT),
        ("E8", E8_IN, E8_OUT),
        ("T", T_IN, T_OUT),
        ("T2", T2_IN, T2_OUT),
        ("T3", T3_IN, T3_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_ff28f65a(inp)
        assert pred == _to_grid(expected), f"{name} failed"

# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "29623171"
SERIAL = "059"
URL    = "https://arcprize.org/play?task=29623171"

# --- Code Golf Concepts ---
CONCEPTS = [
    "detect_grid",
    "separate_images",
    "count_tiles",
    "take_maximum",
    "grid_coloring",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [1, 0, 0, 5, 0, 0, 0, 5, 0, 1, 0],
    [0, 0, 0, 5, 0, 0, 1, 5, 0, 0, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 5, 0, 0, 1, 5, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 1, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 5, 0, 0, 0, 5, 1, 0, 0],
    [0, 1, 0, 5, 0, 0, 0, 5, 0, 0, 1],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 5, 0, 0, 0, 5, 1, 1, 1],
    [0, 0, 0, 5, 0, 0, 0, 5, 1, 1, 1],
    [0, 0, 0, 5, 0, 0, 0, 5, 1, 1, 1],
], dtype=int)

E2_IN = np.array([
    [0, 0, 0, 5, 0, 2, 0, 5, 2, 0, 0],
    [2, 0, 0, 5, 0, 0, 0, 5, 0, 0, 2],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [2, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [2, 0, 0, 5, 0, 0, 2, 5, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 2, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [2, 0, 0, 5, 0, 0, 2, 5, 0, 0, 2],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [0, 0, 0, 5, 0, 0, 0, 5, 2, 2, 2],
    [0, 0, 0, 5, 0, 0, 0, 5, 2, 2, 2],
    [0, 0, 0, 5, 0, 0, 0, 5, 2, 2, 2],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [2, 2, 2, 5, 0, 0, 0, 5, 0, 0, 0],
    [2, 2, 2, 5, 0, 0, 0, 5, 0, 0, 0],
    [2, 2, 2, 5, 0, 0, 0, 5, 0, 0, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
], dtype=int)

E3_IN = np.array([
    [3, 3, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 3, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 3, 0, 5, 0, 3, 0, 5, 0, 0, 0],
    [0, 0, 0, 5, 3, 0, 0, 5, 0, 0, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 3, 0, 5, 3, 0, 0, 5, 3, 3, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 3],
], dtype=int)

E3_OUT = np.array([
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 5, 0, 0, 0, 5, 3, 3, 3],
    [0, 0, 0, 5, 0, 0, 0, 5, 3, 3, 3],
    [0, 0, 0, 5, 0, 0, 0, 5, 3, 3, 3],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [4, 4, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 4, 0, 5, 0, 0, 4, 5, 4, 4, 0],
    [4, 0, 0, 5, 0, 0, 0, 5, 0, 4, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 4, 0],
    [4, 0, 0, 5, 0, 4, 0, 5, 4, 0, 4],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 5, 0, 4, 0, 5, 0, 0, 4],
    [4, 0, 0, 5, 0, 0, 4, 5, 0, 4, 0],
    [0, 0, 0, 5, 4, 4, 0, 5, 0, 0, 0],
], dtype=int)

T_OUT = np.array([
    [4, 4, 4, 5, 0, 0, 0, 5, 0, 0, 0],
    [4, 4, 4, 5, 0, 0, 0, 5, 0, 0, 0],
    [4, 4, 4, 5, 0, 0, 0, 5, 0, 0, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 5, 4, 4, 4, 5, 0, 0, 0],
    [0, 0, 0, 5, 4, 4, 4, 5, 0, 0, 0],
    [0, 0, 0, 5, 4, 4, 4, 5, 0, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j,A=enumerate,c=range(11)):
 E=0;k=[[0 if(i+1)%4>0and(j+1)%4>0 else 5 for i in c]for j in c];W={'00':0,'01':0,'02':0,'10':0,'11':0,'12':0,'20':0,'21':0,'22':0}
 for l,J in A(j):
  for a,C in A(J):
   if C>0 and C!=5:E=int(C);W[str(l//4)+str(a//4)]+=1
 e=max(W.values())
 for l,J in A(k):
  for a,C in A(J):
   if C==0 and W[str(l//4)+str(a//4)]==e:k[l][a]=E
 return k


# --- Code Golf Solution (Compressed) ---
def q(g, n=36, o=0):
    return [[(g[i][j] == 5) * 5 or (sum((S := [g[i & 12 | k % 3][j & 12 | k // 3] for k in R[:9]])) > n != [(o := 1)]) * max(S) for j in R] for i in R] * o or p(g, n - 1)


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

def generate_29623171(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 6))
    w = unifint(diff_lb, diff_ub, (2, 6))
    nh = unifint(diff_lb, diff_ub, (2, 4))
    nw = unifint(diff_lb, diff_ub, (2, 4))
    bgc, linc, fgc = sample(cols, 3)
    fullh = h * nh + (nh - 1)
    fullw = w * nw + (nw - 1)
    c = canvas(linc, (fullh, fullw))
    smallc = canvas(bgc, (h, w))
    inds = totuple(asindices(smallc))
    llocs = set()
    for a in range(0, fullh, h+1):
        for b in range(0, fullw, w + 1):
            llocs.add((a, b))
    llocs = tuple(llocs)
    srcloc = choice(llocs)
    nmostc = unifint(diff_lb, diff_ub, (1, (h * w) // 2 - 1))
    mostc = sample(inds, nmostc)
    srcg = fill(smallc, fgc, mostc)
    obj = asobject(srcg)
    shftd = shift(obj, srcloc)
    gi = paint(c, shftd)
    go = fill(c, fgc, shftd)
    remlocs = remove(srcloc, llocs)
    gg = asobject(fill(smallc, bgc, inds))
    for rl in remlocs:
        noth = unifint(diff_lb, diff_ub, (0, nmostc))
        otherg = fill(smallc, fgc, sample(inds, noth))
        gi = paint(gi, shift(asobject(otherg), rl))
        if noth == nmostc:
            go = fill(go, fgc, shift(obj, rl))
        else:
            go = paint(go, shift(gg, rl))
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Numerical = Union[Integer, IntegerTuple]

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

ContainerContainer = Container[Container]

ZERO = 0

def subtract(
    a: Numerical,
    b: Numerical
) -> Numerical:
    """ subtraction """
    if isinstance(a, int) and isinstance(b, int):
        return a - b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] - b[0], a[1] - b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a - b[0], a - b[1])
    return (a[0] - b, a[1] - b)

def multiply(
    a: Numerical,
    b: Numerical
) -> Numerical:
    """ multiplication """
    if isinstance(a, int) and isinstance(b, int):
        return a * b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] * b[0], a[1] * b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a * b[0], a * b[1])
    return (a[0] * b, a[1] * b)

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

def mfilter(
    container: Container,
    function: Callable
) -> FrozenSet:
    """ filter and merge """
    return merge(sfilter(container, function))

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

def leastcolor(
    element: Element
) -> Integer:
    """ least common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return min(set(values), key=values.count)

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

def toobject(
    patch: Patch,
    grid: Grid
) -> Object:
    """ object from patch and grid """
    h, w = len(grid), len(grid[0])
    return frozenset((grid[i][j], (i, j)) for i, j in toindices(patch) if 0 <= i < h and 0 <= j < w)

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

def replace(
    grid: Grid,
    replacee: Integer,
    replacer: Integer
) -> Grid:
    """ color substitution """
    return tuple(tuple(replacer if v == replacee else v for v in r) for r in grid)

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

def compress(
    grid: Grid
) -> Grid:
    """ removes frontiers from grid """
    ri = tuple(i for i, r in enumerate(grid) if len(set(r)) == 1)
    ci = tuple(j for j, c in enumerate(dmirror(grid)) if len(set(c)) == 1)
    return tuple(tuple(v for j, v in enumerate(r) if j not in ci) for i, r in enumerate(grid) if i not in ri)

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_29623171(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = compress(I)
    x1 = leastcolor(x0)
    x2 = mostcolor(x0)
    x3 = frontiers(I)
    x4 = sfilter(x3, hline)
    x5 = size(x4)
    x6 = increment(x5)
    x7 = sfilter(x3, vline)
    x8 = size(x7)
    x9 = increment(x8)
    x10 = height(I)
    x11 = decrement(x6)
    x12 = subtract(x10, x11)
    x13 = divide(x12, x6)
    x14 = width(I)
    x15 = decrement(x9)
    x16 = subtract(x14, x15)
    x17 = divide(x16, x9)
    x18 = astuple(x13, x17)
    x19 = canvas(ZERO, x18)
    x20 = asindices(x19)
    x21 = astuple(x6, x9)
    x22 = canvas(ZERO, x21)
    x23 = asindices(x22)
    x24 = astuple(x13, x17)
    x25 = increment(x24)
    x26 = rbind(multiply, x25)
    x27 = apply(x26, x23)
    x28 = rbind(toobject, I)
    x29 = lbind(shift, x20)
    x30 = compose(x28, x29)
    x31 = apply(x30, x27)
    x32 = rbind(colorcount, x1)
    x33 = valmax(x31, x32)
    x34 = rbind(colorcount, x1)
    x35 = matcher(x34, x33)
    x36 = mfilter(x31, x35)
    x37 = replace(I, x1, x2)
    x38 = fill(x37, x1, x36)
    return x38


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_29623171(inp)
        assert pred == _to_grid(expected), f"{name} failed"

# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "47c1f68c"
SERIAL = "109"
URL    = "https://arcprize.org/play?task=47c1f68c"

# --- Code Golf Concepts ---
CONCEPTS = [
    "detect_grid",
    "find_the_intruder",
    "crop",
    "recolor",
    "color_guessing",
    "image_repetition",
    "image_reflection",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 0, 0, 2, 0],
    [2, 2, 0, 0, 0, 0, 0, 0, 2, 2],
    [0, 2, 2, 0, 0, 0, 0, 2, 2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 2, 0, 0, 0, 0, 2, 2, 0],
    [2, 2, 0, 0, 0, 0, 0, 0, 2, 2],
    [0, 2, 0, 0, 0, 0, 0, 0, 2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E2_IN = np.array([
    [3, 0, 3, 0, 8, 0, 0, 0, 0],
    [3, 3, 0, 0, 8, 0, 0, 0, 0],
    [3, 0, 0, 0, 8, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 0, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 0, 8, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 0, 0, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [8, 0, 8, 0, 0, 8, 0, 8],
    [8, 8, 0, 0, 0, 0, 8, 8],
    [8, 0, 0, 0, 0, 0, 0, 8],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [8, 0, 0, 0, 0, 0, 0, 8],
    [8, 8, 0, 0, 0, 0, 8, 8],
    [8, 0, 8, 0, 0, 8, 0, 8],
], dtype=int)

E3_IN = np.array([
    [2, 0, 0, 4, 0, 0, 0],
    [0, 2, 2, 4, 0, 0, 0],
    [0, 2, 0, 4, 0, 0, 0],
    [4, 4, 4, 4, 4, 4, 4],
    [0, 0, 0, 4, 0, 0, 0],
    [0, 0, 0, 4, 0, 0, 0],
    [0, 0, 0, 4, 0, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [4, 0, 0, 0, 0, 4],
    [0, 4, 4, 4, 4, 0],
    [0, 4, 0, 0, 4, 0],
    [0, 4, 0, 0, 4, 0],
    [0, 4, 4, 4, 4, 0],
    [4, 0, 0, 0, 0, 4],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 8, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
    [0, 8, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
    [8, 0, 8, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
    [0, 0, 8, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
    [0, 0, 8, 8, 0, 0, 3, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
], dtype=int)

T_OUT = np.array([
    [0, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0],
    [0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0],
    [3, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 3],
    [0, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0],
    [0, 0, 3, 3, 0, 0, 0, 0, 3, 3, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 3, 3, 0, 0, 0, 0, 3, 3, 0, 0],
    [0, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0],
    [3, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 3],
    [0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0],
    [0, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
R=range
L=len
def p(g):
 h,w=L(g),L(g[0])
 C=g[0][w//2]
 X=[[0]*(w-1) for _ in R(h-1)]
 for r in R(h//2):
  for c in R(w//2):
   X[r][c]=g[r][c]
   X[-(r+1)][c]=g[r][c]
   X[-(r+1)][-(c+1)]=g[r][c]
   X[r][-(c+1)]=g[r][c]
 X=[[C if c>0 else 0 for c in r] for r in X]
 return X


# --- Code Golf Solution (Compressed) ---
def q(a, s=0):
    return a * 0 != 0 and (b := [*map(p, a, [a[(l := (len(a) // 2))]] * l)]) + b[::-1] or a % ~a & s


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

Piece = Union[Grid, Patch]

ContainerContainer = Container[Container]

def identity(
    x: Any
) -> Any:
    """ identity function """
    return x

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

def compose(
    outer: Callable,
    inner: Callable
) -> Callable:
    """ function composition """
    return lambda x: outer(inner(x))

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

def recolor(
    value: Integer,
    patch: Patch
) -> Object:
    """ recolor patch """
    return frozenset((value, index) for index in toindices(patch))

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

def hmirror(
    piece: Piece
) -> Piece:
    """ mirroring along horizontal """
    if isinstance(piece, tuple):
        return piece[::-1]
    d = ulcorner(piece)[0] + lrcorner(piece)[0]
    if isinstance(next(iter(piece))[1], tuple):
        return frozenset((v, (d - i, j)) for v, (i, j) in piece)
    return frozenset((d - i, j) for i, j in piece)

def vmirror(
    piece: Piece
) -> Piece:
    """ mirroring along vertical """
    if isinstance(piece, tuple):
        return tuple(row[::-1] for row in piece)
    d = ulcorner(piece)[1] + lrcorner(piece)[1]
    if isinstance(next(iter(piece))[1], tuple):
        return frozenset((v, (i, d - j)) for v, (i, j) in piece)
    return frozenset((i, d - j) for i, j in piece)

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

def replace(
    grid: Grid,
    replacee: Integer,
    replacer: Integer
) -> Grid:
    """ color substitution """
    return tuple(tuple(replacer if v == replacee else v for v in r) for r in grid)

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

def generate_47c1f68c(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(1, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 14))
    w = unifint(diff_lb, diff_ub, (2, 14))
    bgc, linc = sample(cols, 2)
    remcols = difference(cols, (bgc, linc))
    objc = choice(remcols)
    canv = canvas(bgc, (h, w))
    nc = unifint(diff_lb, diff_ub, (1, h * w - 1))
    bx = asindices(canv)
    obj = {choice(totuple(bx))}
    for kk in range(nc - 1):
        dns = mapply(neighbors, obj)
        ch = choice(totuple(bx & dns))
        obj.add(ch)
        bx = bx - {ch}
    obj = recolor(objc, obj)
    gi = paint(canv, obj)
    gi1 = hconcat(hconcat(gi, canvas(linc, (h, 1))), canv)
    gi2 = hconcat(hconcat(canv, canvas(linc, (h, 1))), canv)
    gi = vconcat(vconcat(gi1, canvas(linc, (1, 2*w+1))), gi2)
    go = paint(canv, obj)
    go = hconcat(go, vmirror(go))
    go = vconcat(go, hmirror(go))
    go = replace(go, objc, linc)
    scf = choice((identity, hmirror, vmirror, compose(hmirror, vmirror)))
    gi = scf(gi)
    go = scf(go)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
IntegerSet = FrozenSet[Integer]

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

def combine(
    a: Container,
    b: Container
) -> Container:
    """ union """
    return type(a)((*a, *b))

def argmax(
    container: Container,
    compfunc: Callable
) -> Any:
    """ largest item by custom order """
    return max(container, key=compfunc, default=None)

def astuple(
    a: Integer,
    b: Integer
) -> IntegerTuple:
    """ constructs a tuple """
    return (a, b)

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

def numcolors(
    element: Element
) -> IntegerSet:
    """ number of colors occurring in object or grid """
    return len(palette(element))

def color(
    obj: Object
) -> Integer:
    """ color of object """
    return next(iter(obj))[0]

def rot90(
    grid: Grid
) -> Grid:
    """ quarter clockwise rotation """
    return tuple(row for row in zip(*grid[::-1]))

def rot270(
    grid: Grid
) -> Grid:
    """ quarter anticlockwise rotation """
    return tuple(tuple(row[::-1]) for row in zip(*grid[::-1]))[::-1]

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

def tophalf(
    grid: Grid
) -> Grid:
    """ upper half of grid """
    return grid[:len(grid) // 2]

def bottomhalf(
    grid: Grid
) -> Grid:
    """ lower half of grid """
    return grid[len(grid) // 2 + len(grid) % 2:]

def lefthalf(
    grid: Grid
) -> Grid:
    """ left half of grid """
    return rot270(tophalf(rot90(grid)))

def righthalf(
    grid: Grid
) -> Grid:
    """ right half of grid """
    return rot270(bottomhalf(rot90(grid)))

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

def verify_47c1f68c(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = frontiers(I)
    x1 = merge(x0)
    x2 = color(x1)
    x3 = compress(I)
    x4 = mostcolor(x3)
    x5 = tophalf(I)
    x6 = lefthalf(x5)
    x7 = vmirror(x6)
    x8 = hconcat(x6, x7)
    x9 = hmirror(x8)
    x10 = vconcat(x8, x9)
    x11 = tophalf(I)
    x12 = righthalf(x11)
    x13 = vmirror(x12)
    x14 = hconcat(x13, x12)
    x15 = hmirror(x14)
    x16 = vconcat(x14, x15)
    x17 = bottomhalf(I)
    x18 = lefthalf(x17)
    x19 = vmirror(x18)
    x20 = hconcat(x18, x19)
    x21 = hmirror(x20)
    x22 = vconcat(x21, x20)
    x23 = bottomhalf(I)
    x24 = righthalf(x23)
    x25 = vmirror(x24)
    x26 = hconcat(x25, x24)
    x27 = hmirror(x26)
    x28 = vconcat(x27, x26)
    x29 = astuple(x10, x16)
    x30 = astuple(x22, x28)
    x31 = combine(x29, x30)
    x32 = argmax(x31, numcolors)
    x33 = asindices(x32)
    x34 = ofcolor(x32, x4)
    x35 = difference(x33, x34)
    x36 = fill(x32, x2, x35)
    return x36


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_47c1f68c(inp)
        assert pred == _to_grid(expected), f"{name} failed"

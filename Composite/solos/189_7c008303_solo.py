# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "7c008303"
SERIAL = "189"
URL    = "https://arcprize.org/play?task=7c008303"

# --- Code Golf Concepts ---
CONCEPTS = [
    "color_palette",
    "detect_grid",
    "recoloring",
    "color_guessing",
    "separate_images",
    "crop",
]

# --- Example Grids ---
E1_IN = np.array([
    [2, 4, 8, 0, 0, 0, 0, 0, 0],
    [1, 6, 8, 0, 0, 0, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 8, 0, 3, 0, 0, 3, 0],
    [0, 0, 8, 3, 3, 3, 3, 3, 3],
    [0, 0, 8, 0, 3, 0, 0, 3, 0],
    [0, 0, 8, 0, 3, 0, 0, 3, 0],
    [0, 0, 8, 3, 3, 3, 3, 3, 3],
    [0, 0, 8, 0, 3, 0, 0, 3, 0],
], dtype=int)

E1_OUT = np.array([
    [0, 2, 0, 0, 4, 0],
    [2, 2, 2, 4, 4, 4],
    [0, 2, 0, 0, 4, 0],
    [0, 1, 0, 0, 6, 0],
    [1, 1, 1, 6, 6, 6],
    [0, 1, 0, 0, 6, 0],
], dtype=int)

E2_IN = np.array([
    [0, 0, 0, 0, 0, 0, 8, 1, 2],
    [0, 0, 0, 0, 0, 0, 8, 4, 1],
    [8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 3, 3, 0, 3, 8, 0, 0],
    [3, 3, 0, 0, 0, 0, 8, 0, 0],
    [3, 3, 0, 3, 0, 3, 8, 0, 0],
    [0, 0, 0, 0, 3, 0, 8, 0, 0],
    [3, 3, 3, 3, 3, 3, 8, 0, 0],
    [0, 0, 0, 0, 3, 0, 8, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [0, 0, 1, 2, 0, 2],
    [1, 1, 0, 0, 0, 0],
    [1, 1, 0, 2, 0, 2],
    [0, 0, 0, 0, 1, 0],
    [4, 4, 4, 1, 1, 1],
    [0, 0, 0, 0, 1, 0],
], dtype=int)

E3_IN = np.array([
    [0, 0, 8, 0, 0, 3, 0, 0, 3],
    [0, 0, 8, 0, 0, 3, 0, 0, 3],
    [0, 0, 8, 3, 3, 0, 3, 3, 0],
    [0, 0, 8, 0, 0, 0, 0, 3, 0],
    [0, 0, 8, 0, 3, 0, 3, 0, 0],
    [0, 0, 8, 0, 3, 0, 0, 0, 3],
    [8, 8, 8, 8, 8, 8, 8, 8, 8],
    [2, 4, 8, 0, 0, 0, 0, 0, 0],
    [6, 5, 8, 0, 0, 0, 0, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [0, 0, 2, 0, 0, 4],
    [0, 0, 2, 0, 0, 4],
    [2, 2, 0, 4, 4, 0],
    [0, 0, 0, 0, 5, 0],
    [0, 6, 0, 5, 0, 0],
    [0, 6, 0, 0, 0, 5],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 0, 3, 0, 0, 8, 0, 0],
    [3, 3, 0, 3, 0, 3, 8, 0, 0],
    [0, 3, 0, 3, 0, 3, 8, 0, 0],
    [0, 3, 3, 3, 0, 0, 8, 0, 0],
    [0, 3, 0, 0, 0, 3, 8, 0, 0],
    [0, 0, 3, 0, 0, 0, 8, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 0, 0, 0, 8, 2, 1],
    [0, 0, 0, 0, 0, 0, 8, 4, 7],
], dtype=int)

T_OUT = np.array([
    [0, 0, 0, 1, 0, 0],
    [2, 2, 0, 1, 0, 1],
    [0, 2, 0, 1, 0, 1],
    [0, 4, 4, 7, 0, 0],
    [0, 4, 0, 0, 0, 7],
    [0, 0, 4, 0, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
R=range
L=len
def p(g):
 for i in R(4):
  g=list(map(list,zip(*g[::-1])))
  if g[0][2]==8 and g[2][0]==8:
   for r in R(3,L(g)):
    for c in R(3,L(g[0])):
     if g[r][c]>0:
      g[r][c]=g[(r-2)//4][(c-2)//4]
   g=[r[3:] for r in g[3:]]
 return g


# --- Code Golf Solution (Compressed) ---
def q(g, h=[]):
    return g * 0 != 0 and [*map(p, 3 * [g[(i := (('0' in '%r' % g[2]) * 7))]] + 3 * [g[i + 1]], (h + g)[3 >> i:])] or h % 2 * g


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

def identity(
    x: Any
) -> Any:
    """ identity function """
    return x

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

def rot90(
    grid: Grid
) -> Grid:
    """ quarter clockwise rotation """
    return tuple(row for row in zip(*grid[::-1]))

def rot180(
    grid: Grid
) -> Grid:
    """ half rotation """
    return tuple(tuple(row[::-1]) for row in grid[::-1])

def rot270(
    grid: Grid
) -> Grid:
    """ quarter anticlockwise rotation """
    return tuple(tuple(row[::-1]) for row in zip(*grid[::-1]))[::-1]

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

def generate_7c008303(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 13))
    w = unifint(diff_lb, diff_ub, (2, 13))
    h = h * 2
    w = w * 2
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    linc = choice(remcols)
    remcols = remove(linc, remcols)
    fgc = choice(remcols)
    remcols = remove(fgc, remcols)
    fremcols = sample(remcols, unifint(diff_lb, diff_ub, (1, 4)))
    qc = [choice(fremcols) for j in range(4)]
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    ncd = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    nc = choice((ncd, h * w - ncd))
    nc = min(max(0, nc), h * w)
    cels = sample(inds, nc)
    go = fill(c, fgc, cels)
    gi = canvas(bgc, (h + 3, w + 3))
    gi = paint(gi, shift(asobject(go), (3, 3)))
    gi = fill(gi, linc, connect((2, 0), (2, w + 2)))
    gi = fill(gi, linc, connect((0, 2), (h + 2, 2)))
    gi = fill(gi, qc[0], {(0, 0)})
    gi = fill(gi, qc[1], {(0, 1)})
    gi = fill(gi, qc[2], {(1, 0)})
    gi = fill(gi, qc[3], {(1, 1)})
    A = lefthalf(tophalf(go))
    B = righthalf(tophalf(go))
    C = lefthalf(bottomhalf(go))
    D = righthalf(bottomhalf(go))
    A2 = replace(A, fgc, qc[0])
    B2 = replace(B, fgc, qc[1])
    C2 = replace(C, fgc, qc[2])
    D2 = replace(D, fgc, qc[3])
    go = vconcat(hconcat(A2, B2), hconcat(C2, D2))
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Numerical = Union[Integer, IntegerTuple]

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

ContainerContainer = Container[Container]

F = False

T = True

def halve(
    n: Numerical
) -> Numerical:
    """ scaling by one half """
    return n // 2 if isinstance(n, int) else (n[0] // 2, n[1] // 2)

def contained(
    value: Any,
    container: Container
) -> Boolean:
    """ element of """
    return value in container

def difference(
    a: Container,
    b: Container
) -> Container:
    """ difference """
    return type(a)(e for e in a if e not in b)

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

def argmax(
    container: Container,
    compfunc: Callable
) -> Any:
    """ largest item by custom order """
    return max(container, key=compfunc, default=None)

def argmin(
    container: Container,
    compfunc: Callable
) -> Any:
    """ smallest item by custom order """
    return min(container, key=compfunc, default=None)

def sfilter(
    container: Container,
    condition: Callable
) -> Container:
    """ keep elements in container that satisfy condition """
    return type(container)(e for e in container if condition(e))

def last(
    container: Container
) -> Any:
    """ last item of container """
    return max(enumerate(container))[1]

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

def crop(
    grid: Grid,
    start: IntegerTuple,
    dims: IntegerTuple
) -> Grid:
    """ subgrid specified by start and dimension """
    return tuple(r[start[1]:start[1]+dims[1]] for r in grid[start[0]:start[0]+dims[0]])

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

def hupscale(
    grid: Grid,
    factor: Integer
) -> Grid:
    """ upscale grid horizontally """
    upscaled_grid = tuple()
    for row in grid:
        upscaled_row = tuple()
        for value in row:
            upscaled_row = upscaled_row + tuple(value for num in range(factor))
        upscaled_grid = upscaled_grid + (upscaled_row,)
    return upscaled_grid

def vupscale(
    grid: Grid,
    factor: Integer
) -> Grid:
    """ upscale grid vertically """
    upscaled_grid = tuple()
    for row in grid:
        upscaled_grid = upscaled_grid + tuple(row for num in range(factor))
    return upscaled_grid

def subgrid(
    patch: Patch,
    grid: Grid
) -> Grid:
    """ smallest subgrid containing object """
    return crop(grid, ulcorner(patch), shape(patch))

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

def verify_7c008303(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = frontiers(I)
    x1 = merge(x0)
    x2 = color(x1)
    x3 = shape(I)
    x4 = canvas(x2, x3)
    x5 = hconcat(I, x4)
    x6 = objects(x5, F, F, T)
    x7 = argmin(x6, size)
    x8 = argmax(x6, size)
    x9 = remove(x8, x6)
    x10 = remove(x7, x9)
    x11 = merge(x10)
    x12 = color(x11)
    x13 = subgrid(x8, I)
    x14 = subgrid(x7, I)
    x15 = width(x8)
    x16 = halve(x15)
    x17 = hupscale(x14, x16)
    x18 = height(x8)
    x19 = halve(x18)
    x20 = vupscale(x17, x19)
    x21 = asobject(x20)
    x22 = asindices(x13)
    x23 = ofcolor(x13, x12)
    x24 = difference(x22, x23)
    x25 = rbind(contained, x24)
    x26 = compose(x25, last)
    x27 = sfilter(x21, x26)
    x28 = paint(x13, x27)
    return x28


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_7c008303(inp)
        assert pred == _to_grid(expected), f"{name} failed"

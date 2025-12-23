# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "694f12f3"
SERIAL = "156"
URL    = "https://arcprize.org/play?task=694f12f3"

# --- Code Golf Concepts ---
CONCEPTS = [
    "rectangle_guessing",
    "loop_filling",
    "measure_area",
    "associate_colors_to_ranks",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 4, 4, 4, 4, 0, 0, 0, 0, 0],
    [0, 4, 4, 4, 4, 0, 0, 0, 0, 0],
    [0, 4, 4, 4, 4, 0, 0, 0, 0, 0],
    [0, 4, 4, 4, 4, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 4, 4, 4, 4, 4, 4, 0],
    [0, 0, 0, 4, 4, 4, 4, 4, 4, 0],
    [0, 0, 0, 4, 4, 4, 4, 4, 4, 0],
    [0, 0, 0, 4, 4, 4, 4, 4, 4, 0],
], dtype=int)

E1_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 4, 4, 4, 4, 0, 0, 0, 0, 0],
    [0, 4, 1, 1, 4, 0, 0, 0, 0, 0],
    [0, 4, 1, 1, 4, 0, 0, 0, 0, 0],
    [0, 4, 4, 4, 4, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 4, 4, 4, 4, 4, 4, 0],
    [0, 0, 0, 4, 2, 2, 2, 2, 4, 0],
    [0, 0, 0, 4, 2, 2, 2, 2, 4, 0],
    [0, 0, 0, 4, 4, 4, 4, 4, 4, 0],
], dtype=int)

E2_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 4, 4, 4, 4, 4, 0, 0, 0, 0],
    [0, 4, 4, 4, 4, 4, 0, 0, 0, 0],
    [0, 4, 4, 4, 4, 4, 0, 0, 0, 0],
    [0, 4, 4, 4, 4, 4, 0, 0, 0, 0],
    [0, 4, 4, 4, 4, 4, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 4, 4, 4, 4, 0],
    [0, 0, 0, 0, 0, 4, 4, 4, 4, 0],
    [0, 0, 0, 0, 0, 4, 4, 4, 4, 0],
], dtype=int)

E2_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 4, 4, 4, 4, 4, 0, 0, 0, 0],
    [0, 4, 2, 2, 2, 4, 0, 0, 0, 0],
    [0, 4, 2, 2, 2, 4, 0, 0, 0, 0],
    [0, 4, 2, 2, 2, 4, 0, 0, 0, 0],
    [0, 4, 4, 4, 4, 4, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 4, 4, 4, 4, 0],
    [0, 0, 0, 0, 0, 4, 1, 1, 4, 0],
    [0, 0, 0, 0, 0, 4, 4, 4, 4, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [4, 4, 4, 4, 4, 4, 0, 0, 0, 0],
    [4, 4, 4, 4, 4, 4, 0, 0, 0, 0],
    [4, 4, 4, 4, 4, 4, 0, 0, 0, 0],
    [4, 4, 4, 4, 4, 4, 0, 0, 0, 0],
    [4, 4, 4, 4, 4, 4, 0, 0, 0, 0],
    [4, 4, 4, 4, 4, 4, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 4, 4, 4, 4, 4, 4],
    [0, 0, 0, 0, 4, 4, 4, 4, 4, 4],
    [0, 0, 0, 0, 4, 4, 4, 4, 4, 4],
], dtype=int)

T_OUT = np.array([
    [4, 4, 4, 4, 4, 4, 0, 0, 0, 0],
    [4, 2, 2, 2, 2, 4, 0, 0, 0, 0],
    [4, 2, 2, 2, 2, 4, 0, 0, 0, 0],
    [4, 2, 2, 2, 2, 4, 0, 0, 0, 0],
    [4, 2, 2, 2, 2, 4, 0, 0, 0, 0],
    [4, 4, 4, 4, 4, 4, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 4, 4, 4, 4, 4, 4],
    [0, 0, 0, 0, 4, 1, 1, 1, 1, 4],
    [0, 0, 0, 0, 4, 4, 4, 4, 4, 4],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
R=range
L=len
def p(g):
 h,w,C=L(g),L(g[0]),5
 for r in R(1,h-1):
  if sum(g[r])<1:C+=1
  for c in R(1,w-1):
   if g[r][c] and g[r-1][c] and g[r+1][c] and g[r][c-1] and g[r][c+1]==4:
    g[r][c]=C
 f=sum(g,[])
 Z=sorted([[f.count(c),c] for c in set(f) if c>4])
 for r in R(h):
  for c in R(w):
   if g[r][c]==Z[0][1]:g[r][c]=1
   if g[r][c]==Z[1][1]:g[r][c]=2
 return g


# --- Code Golf Solution (Compressed) ---
def q(g):
    return eval((g := re.sub('(?<=4.{34}4)(?=.{34}4(.*0.{31}(4))?)', "*(X:=g.count)('X\\2X')//X('+')+1", str(g))))


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

def inbox(
    patch: Patch
) -> Indices:
    """ inbox for patch """
    ai, aj = uppermost(patch) + 1, leftmost(patch) + 1
    bi, bj = lowermost(patch) - 1, rightmost(patch) - 1
    si, sj = min(ai, bi), min(aj, bj)
    ei, ej = max(ai, bi), max(aj, bj)
    vlines = {(i, sj) for i in range(si, ei + 1)} | {(i, ej) for i in range(si, ei + 1)}
    hlines = {(si, j) for j in range(sj, ej + 1)} | {(ei, j) for j in range(sj, ej + 1)}
    return frozenset(vlines | hlines)

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

def generate_694f12f3(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2))
    h = unifint(diff_lb, diff_ub, (9, 30))
    w = unifint(diff_lb, diff_ub, (9, 30))
    seploc = randint(4, h - 5)
    bigh = unifint(diff_lb, diff_ub, (4, seploc))
    bigw = unifint(diff_lb, diff_ub, (3, w - 1))
    bigloci = randint(0, seploc - bigh)
    biglocj = randint(0, w - bigw)
    smallmaxh = h - seploc - 1
    smallmaxw = w - 1
    cands = []
    bigsize = bigh * bigw
    for a in range(3, smallmaxh+1):
        for b in range(3, smallmaxw+1):
            if a * b < bigsize:
                cands.append((a, b))
    cands = sorted(cands, key=lambda ab: ab[0]*ab[1])
    num = len(cands)
    idx = unifint(diff_lb, diff_ub, (0, num - 1))
    smallh, smallw = cands[idx]
    smallloci = randint(seploc+1, h - smallh)
    smalllocj = randint(0, w - smallw)
    bgc, sqc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    bigsq = backdrop(frozenset({(bigloci, biglocj), (bigloci + bigh - 1, biglocj + bigw - 1)}))
    smallsq = backdrop(frozenset({(smallloci, smalllocj), (smallloci + smallh - 1, smalllocj + smallw - 1)}))
    gi = fill(gi, sqc, bigsq | smallsq)
    go = fill(gi, 2, backdrop(inbox(bigsq)))
    go = fill(go, 1, backdrop(inbox(smallsq)))
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

ONE = 1

TWO = 2

F = False

T = True

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

def equality(
    a: Any,
    b: Any
) -> Boolean:
    """ equality """
    return a == b

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

def asindices(
    grid: Grid
) -> Indices:
    """ indices of all grid cells """
    return frozenset((i, j) for i in range(len(grid)) for j in range(len(grid[0])))

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

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_694f12f3(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = objects(I, T, F, F)
    x1 = fork(multiply, height, width)
    x2 = fork(equality, size, x1)
    x3 = sfilter(x0, x2)
    x4 = compose(backdrop, inbox)
    x5 = argmin(x3, size)
    x6 = argmax(x3, size)
    x7 = x4(x5)
    x8 = x4(x6)
    x9 = fill(I, ONE, x7)
    x10 = fill(x9, TWO, x8)
    return x10


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_694f12f3(inp)
        assert pred == _to_grid(expected), f"{name} failed"

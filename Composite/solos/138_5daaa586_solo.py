# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "5daaa586"
SERIAL = "138"
URL    = "https://arcprize.org/play?task=5daaa586"

# --- Code Golf Concepts ---
CONCEPTS = [
    "detect_grid",
    "crop",
    "draw_line_from_point",
    "direction_guessing",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0],
    [2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0],
    [2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 8, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 2, 0, 0, 0],
    [0, 0, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 8, 0, 0, 2, 0, 0],
    [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 8, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 0, 3, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 8, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0],
    [2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 8, 2, 2, 2, 2, 2],
    [0, 0, 0, 0, 0, 3, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0],
    [2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0],
    [2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 2, 0],
], dtype=int)

E1_OUT = np.array([
    [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8],
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 8],
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 8],
    [3, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 8],
    [3, 0, 0, 0, 2, 0, 0, 0, 0, 2, 2, 8],
    [3, 0, 0, 0, 2, 0, 0, 0, 0, 2, 2, 8],
    [3, 0, 0, 0, 2, 0, 0, 0, 0, 2, 2, 8],
    [3, 0, 0, 0, 2, 0, 0, 0, 0, 2, 2, 8],
    [3, 0, 2, 0, 2, 0, 0, 0, 0, 2, 2, 8],
    [3, 0, 2, 0, 2, 0, 0, 0, 0, 2, 2, 8],
    [3, 2, 2, 0, 2, 0, 0, 0, 0, 2, 2, 8],
    [3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 8],
], dtype=int)

E2_IN = np.array([
    [0, 0, 4, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 4, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [8, 8, 4, 8, 8, 8, 8, 8, 8, 1, 8, 8],
    [0, 0, 4, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 4, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 4, 0, 0, 0, 8, 0, 0, 1, 0, 8],
    [0, 0, 4, 8, 0, 0, 8, 0, 0, 1, 0, 0],
    [0, 0, 4, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 4, 0, 0, 0, 0, 8, 0, 1, 0, 8],
    [6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 6, 6],
    [0, 0, 4, 0, 0, 0, 8, 0, 0, 1, 0, 0],
    [0, 8, 4, 0, 0, 0, 0, 8, 0, 1, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [4, 8, 8, 8, 8, 8, 8, 1],
    [4, 8, 0, 0, 8, 8, 0, 1],
    [4, 8, 0, 0, 8, 8, 0, 1],
    [4, 8, 0, 0, 8, 8, 0, 1],
    [4, 8, 0, 0, 8, 8, 0, 1],
    [4, 0, 0, 0, 0, 8, 0, 1],
    [4, 0, 0, 0, 0, 8, 0, 1],
    [6, 6, 6, 6, 6, 6, 6, 1],
], dtype=int)

E3_IN = np.array([
    [0, 0, 4, 3, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0],
    [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 0, 4, 0, 0, 0],
    [0, 0, 0, 3, 0, 0, 4, 0, 0, 0, 0, 4, 0, 0, 0],
    [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 0, 4, 0, 0, 0],
    [0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 4, 4, 4, 0, 4],
    [2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2],
    [4, 0, 0, 3, 4, 4, 0, 4, 0, 0, 0, 4, 0, 0, 0],
    [0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0],
    [4, 0, 0, 3, 0, 0, 0, 0, 4, 0, 4, 4, 0, 0, 0],
    [4, 0, 0, 3, 0, 0, 4, 0, 0, 0, 4, 4, 0, 0, 0],
    [8, 8, 8, 3, 8, 8, 8, 8, 8, 8, 8, 4, 8, 8, 8],
    [0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0, 4],
    [0, 0, 0, 3, 4, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0],
    [0, 0, 4, 3, 0, 0, 0, 0, 0, 4, 0, 4, 0, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [3, 2, 2, 2, 2, 2, 2, 2, 4],
    [3, 4, 4, 4, 4, 4, 4, 4, 4],
    [3, 0, 0, 0, 0, 0, 0, 0, 4],
    [3, 0, 0, 0, 0, 4, 4, 4, 4],
    [3, 0, 0, 4, 4, 4, 4, 4, 4],
    [3, 8, 8, 8, 8, 8, 8, 8, 4],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 2, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0],
    [3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3],
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 2, 0, 0],
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 1],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0],
    [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 2, 0, 0],
    [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0],
], dtype=int)

T_OUT = np.array([
    [1, 3, 3, 3, 3, 3, 3, 3, 3, 2],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 2],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 2],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 2],
    [1, 1, 1, 1, 1, 1, 1, 0, 0, 2],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 2],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 2],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(*args, **kwargs):
    raise NotImplementedError("Barnacles solution not available for 138")


# --- Code Golf Solution (Compressed) ---
def q(g, k=59):
    return g * -k or p([[x] + [[r.pop(), x][k - 9 < x in r] for _ in r[0 in g[0]:]] for *r, x in zip(*g)], k - 1)


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, randint, sample, shuffle, uniform

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

def identity(
    x: Any
) -> Any:
    """ identity function """
    return x

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

def vfrontier(
    location: IntegerTuple
) -> Indices:
    """ vertical frontier """
    return frozenset((i, location[1]) for i in range(30))

def hfrontier(
    location: IntegerTuple
) -> Indices:
    """ horizontal frontier """
    return frozenset((location[0], j) for j in range(30))

def shoot(
    start: IntegerTuple,
    direction: IntegerTuple
) -> Indices:
    """ line from starting point and direction """
    return connect(start, (start[0] + 42 * direction[0], start[1] + 42 * direction[1]))

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

def generate_5daaa586(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (7, 30))
    w = unifint(diff_lb, diff_ub, (7, 30))
    loci1 = randint(1, h - 4)
    locj1 = randint(1, w - 4)
    loci1dev = unifint(diff_lb, diff_ub, (0, loci1 - 1))
    locj1dev = unifint(diff_lb, diff_ub, (0, locj1 - 1))
    loci1 -= loci1dev
    locj1 -= locj1dev
    loci2 = unifint(diff_lb, diff_ub, (loci1 + 2, h - 2))
    locj2 = unifint(diff_lb, diff_ub, (locj1 + 2, w - 2))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    c1, c2, c3, c4 = sample(remcols, 4)
    f1 = recolor(c1, hfrontier(toivec(loci1)))
    f2 = recolor(c2, hfrontier(toivec(loci2)))
    f3 = recolor(c3, vfrontier(tojvec(locj1)))
    f4 = recolor(c4, vfrontier(tojvec(locj2)))
    gi = canvas(bgc, (h, w))
    fronts = [f1, f2, f3, f4]
    shuffle(fronts)
    for fr in fronts:
        gi = paint(gi, fr)
    cands = totuple(ofcolor(gi, bgc))
    nn = len(cands)
    nnoise = unifint(diff_lb, diff_ub, (1, max(1, nn // 3)))
    noise = sample(cands, nnoise)
    gi = fill(gi, c1, noise)
    while len(frontiers(gi)) > 4:
        gi = fill(gi, bgc, noise)
        nnoise = unifint(diff_lb, diff_ub, (1, max(1, nn // 3)))
        noise = sample(cands, nnoise)
        if len(set(noise) & ofcolor(gi, c1)) >= len(ofcolor(gi, bgc)):
            break
        gi = fill(gi, c1, noise)
    go = crop(gi, (loci1, locj1), (loci2 - loci1 + 1, locj2 - locj1 + 1))
    ns = ofcolor(go, c1)
    go = fill(go, c1, mapply(rbind(shoot, (-1, 0)), ns))
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

IntegerSet = FrozenSet[Integer]

Element = Union[Object, Grid]

F = False

T = True

UNITY = (1, 1)

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

def difference(
    a: Container,
    b: Container
) -> Container:
    """ difference """
    return type(a)(e for e in a if e not in b)

def argmin(
    container: Container,
    compfunc: Callable
) -> Any:
    """ smallest item by custom order """
    return min(container, key=compfunc, default=None)

def either(
    a: Boolean,
    b: Boolean
) -> Boolean:
    """ logical or """
    return a or b

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

def product(
    a: Container,
    b: Container
) -> FrozenSet:
    """ cartesian product """
    return frozenset((i, j) for j in b for i in a)

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

def colorfilter(
    objs: Objects,
    value: Integer
) -> Objects:
    """ filter objects by color """
    return frozenset(obj for obj in objs if next(iter(obj))[0] == value)

def asindices(
    grid: Grid
) -> Indices:
    """ indices of all grid cells """
    return frozenset((i, j) for i in range(len(grid)) for j in range(len(grid[0])))

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

def bordering(
    patch: Patch,
    grid: Grid
) -> Boolean:
    """ whether a patch is adjacent to a grid border """
    return uppermost(patch) == 0 or leftmost(patch) == 0 or lowermost(patch) == len(grid) - 1 or rightmost(patch) == len(grid[0]) - 1

def palette(
    element: Element
) -> IntegerSet:
    """ colors occurring in object or grid """
    if isinstance(element, tuple):
        return frozenset({v for r in element for v in r})
    return frozenset({v for v, _ in element})

def subgrid(
    patch: Patch,
    grid: Grid
) -> Grid:
    """ smallest subgrid containing object """
    return crop(grid, ulcorner(patch), shape(patch))

def trim(
    grid: Grid
) -> Grid:
    """ trim border of grid """
    return tuple(r[1:-1] for r in grid[1:-1])

def outbox(
    patch: Patch
) -> Indices:
    """ outbox for patch """
    ai, aj = uppermost(patch) - 1, leftmost(patch) - 1
    bi, bj = lowermost(patch) + 1, rightmost(patch) + 1
    si, sj = min(ai, bi), min(aj, bj)
    ei, ej = max(ai, bi), max(aj, bj)
    vlines = {(i, sj) for i in range(si, ei + 1)} | {(i, ej) for i in range(si, ei + 1)}
    hlines = {(si, j) for j in range(sj, ej + 1)} | {(ei, j) for j in range(sj, ej + 1)}
    return frozenset(vlines | hlines)

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_5daaa586(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = mostcolor(I)
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, x0)
    x3 = rbind(bordering, I)
    x4 = compose(flip, x3)
    x5 = mfilter(x2, x4)
    x6 = outbox(x5)
    x7 = subgrid(x6, I)
    x8 = trim(x7)
    x9 = palette(x8)
    x10 = matcher(identity, x0)
    x11 = argmin(x9, x10)
    x12 = trim(x7)
    x13 = ofcolor(x12, x11)
    x14 = shift(x13, UNITY)
    x15 = ofcolor(x7, x11)
    x16 = difference(x15, x14)
    x17 = compose(first, first)
    x18 = compose(first, last)
    x19 = fork(equality, x17, x18)
    x20 = compose(last, first)
    x21 = compose(last, last)
    x22 = fork(equality, x20, x21)
    x23 = fork(either, x19, x22)
    x24 = product(x14, x16)
    x25 = sfilter(x24, x23)
    x26 = fork(connect, first, last)
    x27 = mapply(x26, x25)
    x28 = fill(x7, x11, x27)
    return x28


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_5daaa586(inp)
        assert pred == _to_grid(expected), f"{name} failed"

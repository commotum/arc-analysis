# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "846bdb03"
SERIAL = "201"
URL    = "https://arcprize.org/play?task=846bdb03"

# --- Code Golf Concepts ---
CONCEPTS = [
    "pattern_moving",
    "pattern_reflection",
    "crop",
    "color_matching",
    "x_marks_the_spot",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 2, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 4],
    [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 4],
], dtype=int)

E1_OUT = np.array([
    [4, 0, 0, 0, 0, 0, 0, 4],
    [2, 2, 2, 0, 1, 0, 0, 1],
    [2, 0, 2, 0, 1, 1, 1, 1],
    [2, 0, 2, 2, 1, 0, 0, 1],
    [2, 0, 0, 2, 0, 0, 0, 1],
    [4, 0, 0, 0, 0, 0, 0, 4],
], dtype=int)

E2_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 3, 0, 8, 0, 8, 0, 0, 0],
    [0, 0, 0, 0, 3, 3, 3, 8, 8, 8, 0, 0, 0],
    [0, 0, 0, 0, 0, 3, 0, 8, 0, 8, 0, 0, 0],
    [0, 0, 0, 0, 0, 3, 3, 8, 8, 8, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 8, 0, 8, 0, 0, 0],
    [0, 4, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
    [0, 8, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0],
    [0, 8, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0],
    [0, 8, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0],
    [0, 8, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0],
    [0, 8, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0],
    [0, 4, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [4, 0, 0, 0, 0, 0, 0, 4],
    [8, 8, 0, 8, 0, 3, 0, 3],
    [8, 8, 8, 8, 3, 3, 3, 3],
    [8, 8, 0, 8, 0, 3, 0, 3],
    [8, 8, 8, 8, 3, 3, 0, 3],
    [8, 8, 0, 8, 0, 0, 0, 3],
    [4, 0, 0, 0, 0, 0, 0, 4],
], dtype=int)

E3_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0],
    [0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 2, 2, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [4, 0, 0, 0, 0, 4],
    [2, 0, 2, 1, 1, 1],
    [2, 2, 2, 1, 0, 1],
    [4, 0, 0, 0, 0, 4],
], dtype=int)

E4_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 0, 0],
    [0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 3, 0, 0],
    [0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 3, 0, 0],
    [0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 3, 0, 0],
    [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 7, 7, 0, 3, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 7, 7, 3, 3, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 7, 0, 3, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E4_OUT = np.array([
    [4, 0, 0, 0, 0, 4],
    [7, 7, 7, 0, 3, 3],
    [7, 7, 7, 3, 3, 3],
    [7, 0, 7, 0, 3, 3],
    [4, 0, 0, 0, 0, 4],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 4, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [0, 4, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 8, 0, 8, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 8, 8, 8, 2, 2, 2, 0, 0, 0, 0, 0, 0],
    [0, 8, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
    [0, 8, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0],
], dtype=int)

T_OUT = np.array([
    [4, 0, 0, 0, 0, 0, 0, 4],
    [2, 0, 0, 2, 8, 0, 8, 8],
    [2, 2, 2, 2, 8, 8, 8, 8],
    [2, 0, 2, 0, 0, 0, 8, 8],
    [2, 2, 2, 0, 0, 0, 8, 8],
    [4, 0, 0, 0, 0, 0, 0, 4],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
 A=enumerate;c=next
 E=lambda k,W:k<1 or j[k-1][W]<1 or k>2 and j[k-1][W]==4 and j[k-2][W]>0
 (k,W),(l,J),(a,l),l=[divmod(i,13)for i,v in A(sum(j,[]))if v==4]
 C=c(u for r in zip(*j)if 4not in r for u in r if u)
 e=c(i for i,r in A(j)if any(u==C and E(i,v)for v,u in A(r)))
 K=c(i for i,r in A(zip(*j))if any(u==C and E(v,i)for v,u in A(r)))
 for w in range(a-k-1):
  for L in range(J-W-1):j[k+w+1][[J-L-1,W+L+1][j[k+1][W]==C]],j[e+w][K+L]=j[e+w][K+L],0
 return[r[W:J+1]for r in j[k:a+1]]


# --- Code Golf Solution (Compressed) ---
def q(g):
    f = 1
    m = [R for r in zip(*g) if any((R := [x * f * (f := (f ^ (x == 4) or (g := (g + [x])) > g)) for x in r]))]
    return [(z := [4, *[0] * len(m), 4]), *[[(a := g[15]), *r[::a in m[0] or -1], g[-2]] for *r, in zip(*m) if any(r)], z]


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, randint, sample, uniform

Boolean = bool

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Objects = FrozenSet[Object]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

ContainerContainer = Container[Container]

F = False

T = True

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

def urcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of upper right corner """
    return tuple(map(lambda ix: {0: min, 1: max}[ix[0]](ix[1]), enumerate(zip(*toindices(patch)))))

def llcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of lower left corner """
    return tuple(map(lambda ix: {0: max, 1: min}[ix[0]](ix[1]), enumerate(zip(*toindices(patch)))))

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

def normalize(
    patch: Patch
) -> Patch:
    """ moves upper left corner to origin """
    if len(patch) == 0:
        return patch
    return shift(patch, (-uppermost(patch), -leftmost(patch)))

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

def center(
    patch: Patch
) -> IntegerTuple:
    """ center of the patch """
    return (uppermost(patch) + height(patch) // 2, leftmost(patch) + width(patch) // 2)

def canvas(
    value: Integer,
    dimensions: IntegerTuple
) -> Grid:
    """ grid construction """
    return tuple(tuple(value for j in range(dimensions[1])) for i in range(dimensions[0]))

def corners(
    patch: Patch
) -> Indices:
    """ indices of corners """
    return frozenset({ulcorner(patch), urcorner(patch), llcorner(patch), lrcorner(patch)})

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

def generate_846bdb03(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (12, 30))
    w = unifint(diff_lb, diff_ub, (12, 30))
    oh = unifint(diff_lb, diff_ub, (4, h//2-2))
    ow = unifint(diff_lb, diff_ub, (4, w//2-2))
    bgc, dotc, c1, c2 = sample(cols, 4)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (oh, ow))
    ln1 = connect((1, 0), (oh - 2, 0))
    ln2 = connect((1, ow - 1), (oh - 2, ow - 1))
    go = fill(go, c1, ln1)
    go = fill(go, c2, ln2)
    go = fill(go, dotc, corners(asindices(go)))
    objB = asobject(go)
    bounds = asindices(canvas(-1, (oh - 2, ow - 2)))
    objA = {choice(totuple(bounds))}
    ncells = unifint(diff_lb, diff_ub, (1, ((oh - 2) * (ow - 2)) // 2))
    for k in range(ncells - 1):
        objA.add(choice(totuple((bounds - objA) & mapply(neighbors, objA))))
    while shape(objA) != (oh - 2, ow - 2):
        objA.add(choice(totuple((bounds - objA) & mapply(neighbors, objA))))
    fullinds = asindices(gi)
    loci = randint(0, h - 2 * oh + 2)
    locj = randint(0, w - ow)
    plcdB = shift(objB, (loci, locj))
    plcdi = toindices(plcdB)
    rems = sfilter(fullinds - plcdi, lambda ij: loci + oh <= ij[0] <= h - oh + 2 and ij[1] <= w - ow + 2)
    loc = choice(totuple(rems))
    plcdA = shift(objA, loc)
    mp = center(plcdA)[1]
    plcdAL = sfilter(plcdA, lambda ij: ij[1] < mp)
    plcdAR = plcdA - plcdAL
    plcdA = recolor(c1, plcdAL) | recolor(c2, plcdAR)
    gi = paint(gi, plcdB)
    ism = choice((True, False))
    gi = paint(gi, vmirror(plcdA) if ism else plcdA)
    objA = shift(normalize(plcdA), (1, 1))
    objs = objects(go, T, F, T)
    go = paint(go, objA)
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
IntegerSet = FrozenSet[Integer]

UNITY = (1, 1)

def equality(
    a: Any,
    b: Any
) -> Boolean:
    """ equality """
    return a == b

def greater(
    a: Integer,
    b: Integer
) -> Boolean:
    """ greater """
    return a > b

def size(
    container: Container
) -> Integer:
    """ cardinality """
    return len(container)

def positive(
    x: Integer
) -> Boolean:
    """ positive """
    return x > 0

def extract(
    container: Container,
    condition: Callable
) -> Any:
    """ first element of container that satisfies condition """
    return next(e for e in container if condition(e))

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

def remove(
    value: Any,
    container: Container
) -> Container:
    """ remove item from container """
    return type(container)(e for e in container if e != value)

def branch(
    condition: Boolean,
    if_value: Any,
    else_value: Any
) -> Any:
    """ if else branching """
    return if_value if condition else else_value

def fork(
    outer: Callable,
    a: Callable,
    b: Callable
) -> Callable:
    """ creates a wrapper function """
    return lambda x: outer(a(x), b(x))

def ofcolor(
    grid: Grid,
    value: Integer
) -> Indices:
    """ indices of all grid cells with value """
    return frozenset((i, j) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value)

def crop(
    grid: Grid,
    start: IntegerTuple,
    dims: IntegerTuple
) -> Grid:
    """ subgrid specified by start and dimension """
    return tuple(r[start[1]:start[1]+dims[1]] for r in grid[start[0]:start[0]+dims[0]])

def partition(
    grid: Grid
) -> Objects:
    """ each cell with the same value part of the same object """
    return frozenset(
        frozenset(
            (v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value
        ) for value in palette(grid)
    )

def fgpartition(
    grid: Grid
) -> Objects:
    """ each cell with the same value part of the same object without background """
    return frozenset(
        frozenset(
            (v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value
        ) for value in palette(grid) - {mostcolor(grid)}
    )

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

def subgrid(
    patch: Patch,
    grid: Grid
) -> Grid:
    """ smallest subgrid containing object """
    return crop(grid, ulcorner(patch), shape(patch))

def cover(
    grid: Grid,
    patch: Patch
) -> Grid:
    """ remove object from grid """
    return fill(grid, mostcolor(grid), toindices(patch))

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

def verify_846bdb03(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = partition(I)
    x1 = fork(equality, corners, toindices)
    x2 = extract(x0, x1)
    x3 = subgrid(x2, I)
    x4 = backdrop(x2)
    x5 = cover(I, x4)
    x6 = frontiers(x3)
    x7 = sfilter(x6, hline)
    x8 = size(x7)
    x9 = positive(x8)
    x10 = branch(x9, dmirror, identity)
    x11 = x10(x3)
    x12 = x10(x5)
    x13 = fgpartition(x12)
    x14 = merge(x13)
    x15 = normalize(x14)
    x16 = mostcolor(x12)
    x17 = color(x2)
    x18 = palette(x11)
    x19 = remove(x17, x18)
    x20 = remove(x16, x19)
    x21 = first(x20)
    x22 = last(x20)
    x23 = ofcolor(x11, x22)
    x24 = leftmost(x23)
    x25 = ofcolor(x11, x21)
    x26 = leftmost(x25)
    x27 = greater(x24, x26)
    x28 = ofcolor(x12, x22)
    x29 = leftmost(x28)
    x30 = ofcolor(x12, x21)
    x31 = leftmost(x30)
    x32 = greater(x29, x31)
    x33 = equality(x27, x32)
    x34 = branch(x33, identity, vmirror)
    x35 = x34(x15)
    x36 = shift(x35, UNITY)
    x37 = paint(x11, x36)
    x38 = x10(x37)
    return x38


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("E4", E4_IN, E4_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_846bdb03(inp)
        assert pred == _to_grid(expected), f"{name} failed"

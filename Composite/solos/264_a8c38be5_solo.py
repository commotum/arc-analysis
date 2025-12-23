# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "a8c38be5"
SERIAL = "264"
URL    = "https://arcprize.org/play?task=a8c38be5"

# --- Code Golf Concepts ---
CONCEPTS = [
    "pattern_moving",
    "jigsaw",
    "crop",
]

# --- Example Grids ---
E1_IN = np.array([
    [5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 5, 5, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0],
    [8, 8, 5, 0, 0, 0, 0, 0, 5, 2, 5, 0, 0, 0],
    [0, 0, 2, 5, 5, 0, 0, 0, 5, 5, 5, 0, 0, 0],
    [0, 0, 2, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 2, 5, 5, 0, 5, 5, 5, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 5, 5, 5, 0, 5, 5, 5, 0],
    [0, 5, 1, 1, 0, 0, 5, 5, 5, 0, 5, 4, 5, 0],
    [0, 5, 5, 1, 0, 0, 0, 0, 0, 0, 4, 4, 4, 0],
    [0, 5, 5, 5, 0, 0, 5, 5, 3, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 5, 3, 3, 0, 0, 0, 0, 0],
    [5, 5, 5, 0, 0, 0, 5, 5, 3, 0, 6, 6, 5, 0],
    [5, 5, 9, 0, 0, 0, 0, 0, 0, 0, 6, 5, 5, 0],
    [5, 9, 9, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 0],
], dtype=int)

E1_OUT = np.array([
    [6, 6, 5, 2, 2, 2, 5, 1, 1],
    [6, 5, 5, 5, 2, 5, 5, 5, 1],
    [5, 5, 5, 5, 5, 5, 5, 5, 5],
    [2, 5, 5, 5, 5, 5, 5, 5, 3],
    [2, 2, 5, 5, 5, 5, 5, 3, 3],
    [2, 5, 5, 5, 5, 5, 5, 5, 3],
    [5, 5, 5, 5, 5, 5, 5, 5, 5],
    [8, 5, 5, 5, 4, 5, 5, 5, 9],
    [8, 8, 5, 4, 4, 4, 5, 9, 9],
], dtype=int)

E2_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 4],
    [0, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 5, 4, 4],
    [0, 3, 5, 5, 0, 5, 8, 8, 0, 0, 0, 5, 5, 4],
    [0, 3, 3, 5, 0, 5, 5, 8, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 9, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 9, 9, 0],
    [0, 1, 1, 1, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0],
    [0, 5, 1, 5, 0, 0, 5, 5, 5, 0, 6, 5, 5, 0],
    [0, 5, 5, 5, 0, 0, 5, 5, 5, 0, 6, 6, 5, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 5, 5, 0],
    [0, 0, 0, 0, 7, 7, 5, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 7, 5, 5, 0, 0, 5, 5, 5, 0, 0],
    [0, 0, 0, 0, 5, 5, 5, 0, 0, 5, 2, 5, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [7, 7, 5, 1, 1, 1, 5, 8, 8],
    [7, 5, 5, 5, 1, 5, 5, 5, 8],
    [5, 5, 5, 5, 5, 5, 5, 5, 5],
    [6, 5, 5, 5, 5, 5, 5, 5, 4],
    [6, 6, 5, 5, 5, 5, 5, 4, 4],
    [6, 5, 5, 5, 5, 5, 5, 5, 4],
    [5, 5, 5, 5, 5, 5, 5, 5, 5],
    [3, 5, 5, 5, 2, 5, 5, 5, 9],
    [3, 3, 5, 2, 2, 2, 5, 9, 9],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 0, 0],
    [0, 1, 5, 5, 0, 0, 0, 0, 0, 0, 6, 5, 5, 0, 0],
    [0, 1, 1, 5, 0, 2, 2, 2, 0, 0, 6, 6, 5, 0, 0],
    [0, 1, 5, 5, 0, 5, 2, 5, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 0, 0],
    [0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 5, 5, 5, 0, 0],
    [0, 0, 5, 8, 5, 0, 5, 5, 1, 0, 5, 5, 5, 0, 0],
    [0, 0, 8, 8, 8, 0, 5, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 5, 5, 1, 0, 0, 0, 0, 0, 0],
    [0, 5, 4, 4, 0, 0, 0, 0, 0, 0, 0, 3, 3, 5, 0],
    [0, 5, 5, 4, 0, 0, 0, 0, 0, 0, 0, 3, 5, 5, 0],
    [0, 5, 5, 5, 0, 0, 5, 5, 5, 0, 0, 5, 5, 5, 0],
    [0, 0, 0, 0, 0, 0, 5, 5, 7, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 5, 7, 7, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

T_OUT = np.array([
    [3, 3, 5, 2, 2, 2, 5, 4, 4],
    [3, 5, 5, 5, 2, 5, 5, 5, 4],
    [5, 5, 5, 5, 5, 5, 5, 5, 5],
    [1, 5, 5, 5, 5, 5, 5, 5, 1],
    [1, 1, 5, 5, 5, 5, 5, 1, 1],
    [1, 5, 5, 5, 5, 5, 5, 5, 1],
    [5, 5, 5, 5, 5, 5, 5, 5, 5],
    [6, 5, 5, 5, 8, 5, 5, 5, 7],
    [6, 6, 5, 8, 8, 8, 5, 7, 7],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
L=len
R=range
B=[1,4,9,16,25,36,49,64,81]
S=[-1,0,1]
P=[[x,y] for x in S for y in S]
Z=[264,246,236,194,285,134,156,66,104]
def p(g):
 h,w=L(g),L(g[0])
 X=[[0]*9 for _ in R(9)]
 for r in R(1,h-1):
  for c in R(1,w-1):
    M=[g[r+y][c+x] for y,x in P]
    f=sum([B[i] for i in R(9) if M[i]==5])
    if f in Z and 0 not in M:
     j=Z.index(f)
     for y in R(3):
      for x in R(3):
        X[y+(j//3*3)][x+(j%3*3)]=g[r-1+y][c-1+x]
 return X


# --- Code Golf Solution (Compressed) ---
def q(g):
    return [sum([eval(f'sorted([0in(S:=sum(g,T)),[i!=5for i in S],*g]{'for*g,in map(zip,g,g[1:],g[2:])' * 2})#{g}')[42 >> Y & 7 * ~-X ^ Y][-x] for Y in T], ()) for X in T for x in T]


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

def flip(
    b: Boolean
) -> Boolean:
    """ logical not """
    return not b

def repeat(
    item: Any,
    num: Integer
) -> Tuple:
    """ repetition of item within vector """
    return tuple(item for i in range(num))

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

def bordering(
    patch: Patch,
    grid: Grid
) -> Boolean:
    """ whether a patch is adjacent to a grid border """
    return uppermost(patch) == 0 or leftmost(patch) == 0 or lowermost(patch) == len(grid) - 1 or rightmost(patch) == len(grid[0]) - 1

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

def cmirror(
    piece: Piece
) -> Piece:
    """ mirroring along counterdiagonal """
    if isinstance(piece, tuple):
        return tuple(zip(*(r[::-1] for r in piece[::-1])))
    return vmirror(dmirror(vmirror(piece)))

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

def position(
    a: Patch,
    b: Patch
) -> IntegerTuple:
    """ relative position between two patches """
    ia, ja = center(toindices(a))
    ib, jb = center(toindices(b))
    if ia == ib:
        return (0, 1 if ja < jb else -1)
    elif ja == jb:
        return (1 if ia < ib else -1, 0)
    elif ia < ib:
        return (1, 1 if ja < jb else -1)
    elif ia > ib:
        return (-1, 1 if ja < jb else -1)

def canvas(
    value: Integer,
    dimensions: IntegerTuple
) -> Grid:
    """ grid construction """
    return tuple(tuple(value for j in range(dimensions[1])) for i in range(dimensions[0]))

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

def box(
    patch: Patch
) -> Indices:
    """ outline of patch """
    if len(patch) == 0:
        return patch
    ai, aj = ulcorner(patch)
    bi, bj = lrcorner(patch)
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

def generate_a8c38be5(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    goh = unifint(diff_lb, diff_ub, (9, 20))
    gow = unifint(diff_lb, diff_ub, (9, 20))
    h = unifint(diff_lb, diff_ub, (goh+4, 30))
    w = unifint(diff_lb, diff_ub, (gow+4, 30))
    bgc, sqc = sample(cols, 2)
    remcols = remove(bgc, remove(sqc, cols))
    numc = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, numc)
    go = canvas(sqc, (goh, gow))
    go = fill(go, bgc, box(asindices(go)))
    loci1 = randint(2, goh-7)
    loci2 = randint(loci1+4, goh-3)
    locj1 = randint(2, gow-7)
    locj2 = randint(locj1+4, gow-3)
    f1 = hfrontier((loci1, 0))
    f2 = hfrontier((loci2, 0))
    f3 = vfrontier((0, locj1))
    f4 = vfrontier((0, locj2))
    fs = f1 | f2 | f3 | f4
    go = fill(go, sqc, fs)
    go = fill(go, bgc, {((loci1 + loci2) // 2, 1)})
    go = fill(go, bgc, {((loci1 + loci2) // 2, gow - 2)})
    go = fill(go, bgc, {(1, (locj1 + locj2) // 2)})
    go = fill(go, bgc, {(goh - 2, (locj1 + locj2) // 2)})
    objs = objects(go, T, F, T)
    objs = merge(set(recolor(choice(ccols), obj) for obj in objs))
    go = paint(go, objs)
    gi = go
    hdelt = h - goh
    hdelt1 = randint(1, hdelt - 3)
    hdelt2 = randint(1, hdelt - hdelt1 - 2)
    hdelt3 = randint(1, hdelt - hdelt1 - hdelt2 - 1)
    hdelt4 = hdelt - hdelt1 - hdelt2 - hdelt3
    wdelt = w - gow
    wdelt1 = randint(1, wdelt - 3)
    wdelt2 = randint(1, wdelt - wdelt1 - 2)
    wdelt3 = randint(1, wdelt - wdelt1 - wdelt2 - 1)
    wdelt4 = wdelt - wdelt1 - wdelt2 - wdelt3
    gi = gi[:loci2] + repeat(repeat(bgc, gow), hdelt2) + gi[loci2:]
    gi = gi[:loci1+1] + repeat(repeat(bgc, gow), hdelt3) + gi[loci1+1:]
    gi = repeat(repeat(bgc, gow), hdelt1) + gi + repeat(repeat(bgc, gow), hdelt4)
    gi = dmirror(gi)
    gi = gi[:locj2] + repeat(repeat(bgc, h), wdelt2) + gi[locj2:]
    gi = gi[:locj1+1] + repeat(repeat(bgc, h), wdelt3) + gi[locj1+1:]
    gi = repeat(repeat(bgc, h), wdelt1) + gi + repeat(repeat(bgc, h), wdelt4)
    gi = dmirror(gi)
    nswitcheroos = unifint(diff_lb, diff_ub, (0, 10))
    if choice((True, False)):
        gi = gi[loci1+hdelt1+1:] + gi[:loci1+hdelt1+1]
    if choice((True, False)):
        gi = dmirror(gi)
        gi = gi[locj1+wdelt1+1:] + gi[:locj1+wdelt1+1]
        gi = dmirror(gi)
    for k in range(nswitcheroos):
        o = asobject(gi)
        tmpc = canvas(bgc, (h+12, w+12))
        tmpc = paint(tmpc, shift(o, (6, 6)))
        objs = objects(tmpc, F, T, T)
        objs = apply(rbind(shift, (-6, -6)), objs)
        mpr = dict()
        for obj in objs:
            shp = shape(obj)
            if shp in mpr:
                mpr[shp].append(obj)
            else:
                mpr[shp] = [obj]
        if max([len(x) for x in mpr.values()]) == 1:
            break
        ress = [(kk, v) for kk, v in mpr.items() if len(v) > 1]
        res, abc = choice(ress)
        a, b = sample(abc, 2)
        ulca = ulcorner(a)
        ulcb = ulcorner(b)
        ap = shift(normalize(a), ulcb)
        bp = shift(normalize(b), ulca)
        gi = paint(gi, ap | bp)
    nshifts = unifint(diff_lb, diff_ub, (0, 30))
    for k in range(nshifts):
        o = asobject(gi)
        tmpc = canvas(bgc, (h+12, w+12))
        tmpc = paint(tmpc, shift(o, (6, 6)))
        objs = objects(tmpc, F, F, T)
        objs = apply(rbind(shift, (-6, -6)), objs)
        objs = sfilter(objs, compose(flip, rbind(bordering, gi)))
        if len(objs) == 0:
            break
        obj = choice(totuple(objs))
        direc1 = (randint(-1, 1), randint(-1, 1))
        direc2 = position({(h//2, w//2)}, {center(obj)})
        direc = choice((direc1, direc2))
        gi = fill(gi, bgc, obj)
        gi = paint(gi, shift(obj, direc))
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Numerical = Union[Integer, IntegerTuple]

IntegerSet = FrozenSet[Integer]

ZERO = 0

ONE = 1

TWO = 2

def add(
    a: Numerical,
    b: Numerical
) -> Numerical:
    """ addition """
    if isinstance(a, int) and isinstance(b, int):
        return a + b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] + b[0], a[1] + b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a + b[0], a + b[1])
    return (a[0] + b, a[1] + b)

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

def equality(
    a: Any,
    b: Any
) -> Boolean:
    """ equality """
    return a == b

def contained(
    value: Any,
    container: Container
) -> Boolean:
    """ element of """
    return value in container

def combine(
    a: Container,
    b: Container
) -> Container:
    """ union """
    return type(a)((*a, *b))

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

def argmax(
    container: Container,
    compfunc: Callable
) -> Any:
    """ largest item by custom order """
    return max(container, key=compfunc, default=None)

def both(
    a: Boolean,
    b: Boolean
) -> Boolean:
    """ logical and """
    return a and b

def increment(
    x: Numerical
) -> Numerical:
    """ incrementing """
    return x + 1 if isinstance(x, int) else (x[0] + 1, x[1] + 1)

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

def chain(
    h: Callable,
    g: Callable,
    f: Callable
) -> Callable:
    """ function composition with three functions """
    return lambda x: h(g(f(x)))

def matcher(
    function: Callable,
    target: Any
) -> Callable:
    """ construction of equality function """
    return lambda x: function(x) == target

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

def fork(
    outer: Callable,
    a: Callable,
    b: Callable
) -> Callable:
    """ creates a wrapper function """
    return lambda x: outer(a(x), b(x))

def colorcount(
    element: Element,
    value: Integer
) -> Integer:
    """ number of cells with color """
    if isinstance(element, tuple):
        return sum(row.count(value) for row in element)
    return sum(v == value for v, _ in element)

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

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_a8c38be5(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = objects(I, T, F, F)
    x1 = mostcolor(I)
    x2 = palette(I)
    x3 = remove(x1, x2)
    x4 = lbind(colorcount, I)
    x5 = argmax(x3, x4)
    x6 = astuple(x1, x5)
    x7 = rbind(contained, x6)
    x8 = chain(flip, x7, color)
    x9 = sfilter(x0, x8)
    x10 = fork(connect, ulcorner, urcorner)
    x11 = fork(connect, ulcorner, llcorner)
    x12 = fork(combine, x10, x11)
    x13 = fork(equality, toindices, x12)
    x14 = fork(connect, urcorner, ulcorner)
    x15 = fork(connect, urcorner, lrcorner)
    x16 = fork(combine, x14, x15)
    x17 = fork(equality, toindices, x16)
    x18 = fork(connect, llcorner, ulcorner)
    x19 = fork(connect, llcorner, lrcorner)
    x20 = fork(combine, x18, x19)
    x21 = fork(equality, toindices, x20)
    x22 = fork(connect, lrcorner, llcorner)
    x23 = fork(connect, lrcorner, urcorner)
    x24 = fork(combine, x22, x23)
    x25 = fork(equality, toindices, x24)
    x26 = fork(contained, lrcorner, toindices)
    x27 = compose(flip, x26)
    x28 = fork(contained, llcorner, toindices)
    x29 = compose(flip, x28)
    x30 = fork(contained, urcorner, toindices)
    x31 = compose(flip, x30)
    x32 = fork(contained, ulcorner, toindices)
    x33 = compose(flip, x32)
    x34 = fork(both, x27, x29)
    x35 = fork(both, x31, x33)
    x36 = fork(both, x31, x27)
    x37 = fork(both, x33, x29)
    x38 = lbind(matcher, first)
    x39 = compose(x38, lowermost)
    x40 = fork(sfilter, toindices, x39)
    x41 = compose(size, x40)
    x42 = matcher(x41, ONE)
    x43 = lbind(matcher, first)
    x44 = compose(x43, uppermost)
    x45 = fork(sfilter, toindices, x44)
    x46 = compose(size, x45)
    x47 = matcher(x46, ONE)
    x48 = lbind(matcher, last)
    x49 = compose(x48, rightmost)
    x50 = fork(sfilter, toindices, x49)
    x51 = compose(size, x50)
    x52 = matcher(x51, ONE)
    x53 = lbind(matcher, last)
    x54 = compose(x53, leftmost)
    x55 = fork(sfilter, toindices, x54)
    x56 = compose(size, x55)
    x57 = matcher(x56, ONE)
    x58 = fork(both, x34, x42)
    x59 = fork(both, x35, x47)
    x60 = fork(both, x36, x52)
    x61 = fork(both, x37, x57)
    x62 = fork(connect, ulcorner, urcorner)
    x63 = fork(difference, x62, toindices)
    x64 = compose(size, x63)
    x65 = matcher(x64, ZERO)
    x66 = fork(connect, llcorner, lrcorner)
    x67 = fork(difference, x66, toindices)
    x68 = compose(size, x67)
    x69 = matcher(x68, ZERO)
    x70 = fork(connect, ulcorner, llcorner)
    x71 = fork(difference, x70, toindices)
    x72 = compose(size, x71)
    x73 = matcher(x72, ZERO)
    x74 = fork(connect, urcorner, lrcorner)
    x75 = fork(difference, x74, toindices)
    x76 = compose(size, x75)
    x77 = matcher(x76, ZERO)
    x78 = fork(both, x65, x58)
    x79 = fork(both, x69, x59)
    x80 = fork(both, x73, x60)
    x81 = fork(both, x77, x61)
    x82 = argmax(x9, x13)
    x83 = argmax(x9, x17)
    x84 = argmax(x9, x21)
    x85 = argmax(x9, x25)
    x86 = argmax(x9, x78)
    x87 = argmax(x9, x79)
    x88 = argmax(x9, x80)
    x89 = argmax(x9, x81)
    x90 = height(x82)
    x91 = height(x84)
    x92 = add(x90, x91)
    x93 = height(x88)
    x94 = add(x93, TWO)
    x95 = add(x92, x94)
    x96 = width(x82)
    x97 = width(x83)
    x98 = add(x96, x97)
    x99 = width(x86)
    x100 = add(x99, TWO)
    x101 = add(x98, x100)
    x102 = ulcorner(x82)
    x103 = increment(x102)
    x104 = index(I, x103)
    x105 = astuple(x95, x101)
    x106 = canvas(x104, x105)
    x107 = normalize(x82)
    x108 = paint(x106, x107)
    x109 = normalize(x83)
    x110 = width(x83)
    x111 = subtract(x101, x110)
    x112 = tojvec(x111)
    x113 = shift(x109, x112)
    x114 = paint(x108, x113)
    x115 = normalize(x84)
    x116 = height(x84)
    x117 = subtract(x95, x116)
    x118 = toivec(x117)
    x119 = shift(x115, x118)
    x120 = paint(x114, x119)
    x121 = normalize(x85)
    x122 = height(x85)
    x123 = subtract(x95, x122)
    x124 = width(x85)
    x125 = subtract(x101, x124)
    x126 = astuple(x123, x125)
    x127 = shift(x121, x126)
    x128 = paint(x120, x127)
    x129 = normalize(x88)
    x130 = height(x82)
    x131 = increment(x130)
    x132 = toivec(x131)
    x133 = shift(x129, x132)
    x134 = paint(x128, x133)
    x135 = normalize(x86)
    x136 = width(x82)
    x137 = increment(x136)
    x138 = tojvec(x137)
    x139 = shift(x135, x138)
    x140 = paint(x134, x139)
    x141 = normalize(x89)
    x142 = height(x83)
    x143 = increment(x142)
    x144 = width(x89)
    x145 = subtract(x101, x144)
    x146 = astuple(x143, x145)
    x147 = shift(x141, x146)
    x148 = paint(x140, x147)
    x149 = normalize(x87)
    x150 = height(x87)
    x151 = subtract(x95, x150)
    x152 = width(x84)
    x153 = increment(x152)
    x154 = astuple(x151, x153)
    x155 = shift(x149, x154)
    x156 = paint(x148, x155)
    return x156


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_a8c38be5(inp)
        assert pred == _to_grid(expected), f"{name} failed"

# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "de1cd16c"
SERIAL = "355"
URL    = "https://arcprize.org/play?task=de1cd16c"

# --- Code Golf Concepts ---
CONCEPTS = [
    "separate_images",
    "count_tiles",
    "take_maximum",
    "summarize",
]

# --- Example Grids ---
E1_IN = np.array([
    [4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0],
    [4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0],
    [4, 4, 4, 6, 4, 4, 4, 4, 0, 0, 6, 0, 0],
    [4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0],
    [4, 4, 4, 4, 4, 6, 4, 4, 0, 0, 0, 0, 0],
    [4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0],
    [4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 1, 1, 1, 1, 1],
    [8, 8, 8, 8, 8, 8, 8, 8, 1, 1, 1, 1, 1],
    [8, 8, 8, 8, 8, 8, 8, 8, 1, 1, 1, 1, 1],
    [8, 8, 8, 8, 8, 6, 8, 8, 1, 1, 1, 1, 1],
    [8, 8, 8, 8, 8, 8, 8, 8, 1, 1, 1, 6, 1],
    [8, 8, 6, 8, 8, 8, 8, 8, 1, 1, 1, 1, 1],
    [8, 8, 8, 8, 8, 8, 8, 8, 1, 1, 1, 1, 1],
    [8, 8, 8, 8, 8, 8, 8, 8, 1, 1, 1, 1, 1],
    [8, 8, 8, 8, 8, 6, 8, 8, 1, 1, 1, 1, 1],
    [8, 8, 8, 8, 8, 8, 8, 8, 1, 1, 1, 1, 1],
], dtype=int)

E1_OUT = np.array([
    [8],
], dtype=int)

E2_IN = np.array([
    [3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2],
    [3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 1, 2, 2, 1, 2],
    [3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2],
    [3, 3, 3, 3, 3, 3, 3, 2, 2, 1, 2, 2, 2, 2, 2],
    [3, 3, 1, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2],
    [3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2],
    [3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 1, 2, 2, 2],
    [3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2],
    [3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
], dtype=int)

E2_OUT = np.array([
    [2],
], dtype=int)

E3_IN = np.array([
    [1, 1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [1, 4, 1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [1, 1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [1, 1, 1, 4, 1, 1, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [1, 1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [1, 1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 4, 5, 5, 5, 5, 5],
    [1, 1, 4, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [1, 1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    [0, 0, 0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 4, 6, 6, 6, 6],
    [0, 0, 0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    [0, 0, 0, 0, 0, 0, 0, 6, 6, 6, 4, 6, 6, 6, 6, 4, 6],
    [0, 0, 0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    [0, 0, 0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    [0, 0, 0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6, 4, 6, 6, 6],
    [0, 0, 0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
], dtype=int)

E3_OUT = np.array([
    [6],
], dtype=int)

E4_IN = np.array([
    [1, 1, 1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [1, 1, 1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [1, 1, 1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [1, 1, 1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [1, 1, 1, 1, 1, 1, 1, 8, 8, 8, 8, 2, 8, 8, 8, 8, 8, 8, 8],
    [1, 1, 1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [1, 1, 1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [1, 1, 1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [1, 1, 1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [1, 1, 1, 1, 1, 1, 1, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4],
    [1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
], dtype=int)

E4_OUT = np.array([
    [4],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [2, 4, 2, 2, 2, 2, 2, 4, 2, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [2, 2, 2, 2, 4, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [2, 2, 4, 2, 2, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [2, 2, 2, 2, 2, 2, 4, 2, 2, 8, 8, 8, 8, 8, 4, 8, 8, 8, 8],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 8, 8, 8, 4, 8, 8, 8, 8, 8, 8],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 4, 8, 8, 8, 8],
    [1, 1, 1, 1, 4, 1, 1, 1, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [1, 4, 1, 1, 1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
], dtype=int)

T_OUT = np.array([
    [2],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
L=len
R=range
E=enumerate
def M(m,C,Z):
 P=[[x,y] for y,r in E(m) for x,c in E(r) if c==C]
 f=sum(P,[]);x=f[::2];y=f[1::2]
 X=m[min(y):max(y)+1]
 X=[r[min(x):max(x)+1] for r in X]
 return sum(X,[]).count(Z)
def p(g):
 f=sum(g,[]);Z=sorted([[f.count(k),k] for k in set(f)])
 Z=[x[1] for x in Z]
 P=[M(g,Z[i],Z[0]) for i in R(1,L(Z))]
 return [[Z[P.index(max(P))+1]]]


# --- Code Golf Solution (Compressed) ---
def q(g):
    return [sorted({*(A := [(*{*j} & {*a} ^ {b},) for j in g for *a, b in zip(*g, j)])}, key=A.count)[-3]]


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

F = False

T = True

def order(
    container: Container,
    compfunc: Callable
) -> Tuple:
    """ order container by custom key """
    return tuple(sorted(container, key=compfunc))

def size(
    container: Container
) -> Integer:
    """ cardinality """
    return len(container)

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

def generate_de1cd16c(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    noisec = choice(cols)
    remcols = remove(noisec, cols)
    ncols = unifint(diff_lb, diff_ub, (2, 9))
    ccols = sample(remcols, ncols)
    starterc = ccols[0]
    ccols = ccols[1:]
    gi = canvas(starterc, (h, w))
    for k in range(ncols - 1):
        objs = objects(gi, T, F, F)
        objs = sfilter(objs, lambda o: height(o) > 5 or width(o) > 5)
        if len(objs) == 0:
            break
        objs = totuple(objs)
        obj = choice(objs)
        if height(obj) > 5 and width(obj) > 5:
            ax = choice((0, 1))
        elif height(obj) > 5:
            ax = 0
        elif width(obj) > 5:
            ax = 1
        if ax == 0:
            loci = randint(uppermost(obj)+3, lowermost(obj)-2)
            newobj = sfilter(toindices(obj), lambda ij: ij[0] >= loci)
        elif ax == 1:
            locj = randint(leftmost(obj)+3, rightmost(obj)-2)
            newobj = sfilter(toindices(obj), lambda ij: ij[1] >= locj)
        gi = fill(gi, ccols[k], newobj)
    objs = order(objects(gi, T, F, F), size)
    allowances = [max(1, ((height(o) - 2) * (width(o) - 2)) // 2) for o in objs]
    meann = max(1, int(sum(allowances) / len(allowances)))
    chosens = [randint(0, min(meann, allowed)) for allowed in allowances]
    while max(chosens) == 0:
        chosens = [randint(0, min(meann, allowed)) for allowed in allowances]
    mx = max(chosens)
    fixinds = [idx for idx, cnt in enumerate(chosens) if cnt == mx]
    gogoind = fixinds[0]
    gogocol = color(objs[gogoind])
    fixinds = fixinds[1:]
    for idx in fixinds:
        chosens[idx] -= 1
    for obj, cnt in zip(objs, chosens):
        locs = sample(totuple(backdrop(inbox(toindices(obj)))), cnt)
        gi = fill(gi, noisec, locs)
    go = canvas(gogocol, (1, 1))
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
ContainerContainer = Container[Container]

UNITY = (1, 1)

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

def dedupe(
    iterable: Tuple
) -> Tuple:
    """ remove duplicates """
    return tuple(e for i, e in enumerate(iterable) if iterable.index(e) == i)

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

def mostcommon(
    container: Container
) -> Any:
    """ most common item """
    return max(set(container), key=container.count)

def branch(
    condition: Boolean,
    if_value: Any,
    else_value: Any
) -> Any:
    """ if else branching """
    return if_value if condition else else_value

def compose(
    outer: Callable,
    inner: Callable
) -> Callable:
    """ function composition """
    return lambda x: outer(inner(x))

def chain(
    h: Callable,
    g: Callable,
    f: Callable
) -> Callable:
    """ function composition with three functions """
    return lambda x: h(g(f(x)))

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

def leastcolor(
    element: Element
) -> Integer:
    """ least common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return min(set(values), key=values.count)

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

def subgrid(
    patch: Patch,
    grid: Grid
) -> Grid:
    """ smallest subgrid containing object """
    return crop(grid, ulcorner(patch), shape(patch))

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_de1cd16c(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = objects(I, T, F, F)
    x1 = totuple(x0)
    x2 = apply(color, x1)
    x3 = size(x2)
    x4 = dedupe(x2)
    x5 = size(x4)
    x6 = equality(x3, x5)
    x7 = compose(leastcolor, merge)
    x8 = lbind(apply, color)
    x9 = chain(mostcommon, x8, totuple)
    x10 = branch(x6, x7, x9)
    x11 = x10(x0)
    x12 = objects(I, T, F, F)
    x13 = colorfilter(x12, x11)
    x14 = difference(x12, x13)
    x15 = rbind(subgrid, I)
    x16 = apply(x15, x14)
    x17 = rbind(colorcount, x11)
    x18 = argmax(x16, x17)
    x19 = mostcolor(x18)
    x20 = canvas(x19, UNITY)
    return x20


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("E4", E4_IN, E4_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_de1cd16c(inp)
        assert pred == _to_grid(expected), f"{name} failed"

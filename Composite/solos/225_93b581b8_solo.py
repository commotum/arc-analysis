# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "93b581b8"
SERIAL = "225"
URL    = "https://arcprize.org/play?task=93b581b8"

# --- Code Golf Concepts ---
CONCEPTS = [
    "pattern_expansion",
    "color_guessing",
    "out_of_boundary",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 9, 3, 0, 0],
    [0, 0, 7, 8, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [8, 8, 0, 0, 7, 7],
    [8, 8, 0, 0, 7, 7],
    [0, 0, 9, 3, 0, 0],
    [0, 0, 7, 8, 0, 0],
    [3, 3, 0, 0, 9, 9],
    [3, 3, 0, 0, 9, 9],
], dtype=int)

E2_IN = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 4, 6, 0, 0, 0],
    [0, 2, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [1, 0, 0, 2, 2, 0],
    [0, 4, 6, 0, 0, 0],
    [0, 2, 1, 0, 0, 0],
    [6, 0, 0, 4, 4, 0],
    [6, 0, 0, 4, 4, 0],
    [0, 0, 0, 0, 0, 0],
], dtype=int)

E3_IN = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 3, 6, 0, 0],
    [0, 0, 5, 2, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [2, 2, 0, 0, 5, 5],
    [2, 2, 0, 0, 5, 5],
    [0, 0, 3, 6, 0, 0],
    [0, 0, 5, 2, 0, 0],
    [6, 6, 0, 0, 3, 3],
    [6, 6, 0, 0, 3, 3],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 3, 1, 0, 0],
    [0, 0, 2, 5, 0, 0],
    [0, 0, 0, 0, 0, 0],
], dtype=int)

T_OUT = np.array([
    [0, 0, 0, 0, 0, 0],
    [5, 5, 0, 0, 2, 2],
    [5, 5, 0, 0, 2, 2],
    [0, 0, 3, 1, 0, 0],
    [0, 0, 2, 5, 0, 0],
    [1, 1, 0, 0, 3, 3],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(g,L=len,R=range):
 h,w=L(g),L(g[0])
 r=[r for r in R(h) if L(set(g[r]))>1][0]
 c=[c for c in R(w) if g[r][c]>0][0]
 P=[[-2,-2,g[r+1][c+1]],[2,-2,g[r][c+1]],[-2,2,g[r+1][c]],[2,2,g[r][c]]]
 for i in R(r,r+2):
  for j in R(c,c+2):
   for y,x,C in P:
    if 0<=y+i<h and 0<=x+j<w:
     g[y+i][x+j]=C
 return g


# --- Code Golf Solution (Compressed) ---
def q(*args, **kwargs):
    return (eval('lambda g:[[[g[y][x]|g[Y-V[y-Y>>1]][X-V[x-X>>1]]' + 'for %s in range(6)%s' * 4 % (*'x]y]Y X', 'if g[Y][X]][0]')))(*args, **kwargs)


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

ContainerContainer = Container[Container]

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

def dneighbors(
    loc: IntegerTuple
) -> Indices:
    """ directly adjacent indices """
    return frozenset({(loc[0] - 1, loc[1]), (loc[0] + 1, loc[1]), (loc[0], loc[1] - 1), (loc[0], loc[1] + 1)})

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

def generate_93b581b8(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(remcols, numcols)
    numocc = unifint(diff_lb, diff_ub, (1, (h * w) // 50))
    succ = 0
    tr = 0
    maxtr = 10 * numocc
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    fullinds = asindices(gi)
    while tr < maxtr and succ < numocc:
        tr += 1
        cands = sfilter(inds, lambda ij: ij[0] <= h - 2 and ij[1] <= w - 2)
        if len(cands) == 0:
            break
        loc = choice(totuple(cands))
        c1, c2, c3, c4 = [choice(ccols) for k in range(4)]
        q = {(0, 0), (0, 1), (1, 0), (1, 1)}
        inobj = {(c1, (0, 0)), (c2, (0, 1)), (c3, (1, 0)), (c4, (1, 1))}
        outobj = inobj | recolor(c4, shift(q, (-2, -2))) | recolor(c3, shift(q, (-2, 2))) | recolor(c2, shift(q, (2, -2))) | recolor(c1, shift(q, (2, 2)))
        inobjplcd = shift(inobj, loc)
        outobjplcd = shift(outobj, loc)
        outobjplcd = sfilter(outobjplcd, lambda cij: cij[1] in fullinds)
        outobjplcdi = toindices(outobjplcd)
        if outobjplcdi.issubset(inds):
            succ += 1
            inds = (inds - outobjplcdi) - mapply(dneighbors, toindices(inobjplcd))
            gi = paint(gi, inobjplcd)
            go = paint(go, outobjplcd)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

TWO = 2

F = False

T = True

NEG_TWO = -2

TWO_BY_TWO = (2, 2)

def combine(
    a: Container,
    b: Container
) -> Container:
    """ union """
    return type(a)((*a, *b))

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

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_93b581b8(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = objects(I, F, F, T)
    x1 = apply(toindices, x0)
    x2 = lbind(index, I)
    x3 = compose(x2, lrcorner)
    x4 = astuple(NEG_TWO, NEG_TWO)
    x5 = rbind(shift, x4)
    x6 = fork(recolor, x3, x5)
    x7 = compose(x2, ulcorner)
    x8 = rbind(shift, TWO_BY_TWO)
    x9 = fork(recolor, x7, x8)
    x10 = compose(x2, llcorner)
    x11 = astuple(NEG_TWO, TWO)
    x12 = rbind(shift, x11)
    x13 = fork(recolor, x10, x12)
    x14 = compose(x2, urcorner)
    x15 = astuple(TWO, NEG_TWO)
    x16 = rbind(shift, x15)
    x17 = fork(recolor, x14, x16)
    x18 = fork(combine, x6, x9)
    x19 = fork(combine, x13, x17)
    x20 = fork(combine, x18, x19)
    x21 = mapply(x20, x1)
    x22 = paint(I, x21)
    return x22


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_93b581b8(inp)
        assert pred == _to_grid(expected), f"{name} failed"

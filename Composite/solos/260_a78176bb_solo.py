# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "a78176bb"
SERIAL = "260"
URL    = "https://arcprize.org/play?task=a78176bb"

# --- Code Golf Concepts ---
CONCEPTS = [
    "draw_parallel_line",
    "direction_guessing",
    "remove_intruders",
]

# --- Example Grids ---
E1_IN = np.array([
    [7, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 7, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 7, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 7, 5, 5, 0, 0, 0, 0],
    [0, 0, 0, 0, 7, 5, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 7, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 7, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 7, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 7, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 7],
], dtype=int)

E1_OUT = np.array([
    [7, 0, 0, 0, 7, 0, 0, 0, 0, 0],
    [0, 7, 0, 0, 0, 7, 0, 0, 0, 0],
    [0, 0, 7, 0, 0, 0, 7, 0, 0, 0],
    [0, 0, 0, 7, 0, 0, 0, 7, 0, 0],
    [0, 0, 0, 0, 7, 0, 0, 0, 7, 0],
    [0, 0, 0, 0, 0, 7, 0, 0, 0, 7],
    [0, 0, 0, 0, 0, 0, 7, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 7, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 7, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 7],
], dtype=int)

E2_IN = np.array([
    [0, 0, 0, 0, 0, 9, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 5, 9, 0, 0, 0],
    [0, 0, 0, 0, 0, 5, 5, 9, 0, 0],
    [0, 0, 0, 0, 0, 5, 5, 5, 9, 0],
    [0, 0, 0, 0, 0, 5, 5, 5, 5, 9],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [0, 0, 0, 0, 0, 9, 0, 0, 0, 0],
    [9, 0, 0, 0, 0, 0, 9, 0, 0, 0],
    [0, 9, 0, 0, 0, 0, 0, 9, 0, 0],
    [0, 0, 9, 0, 0, 0, 0, 0, 9, 0],
    [0, 0, 0, 9, 0, 0, 0, 0, 0, 9],
    [0, 0, 0, 0, 9, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 9, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 9, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 9, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 9, 0],
], dtype=int)

E3_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 2, 5, 5, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 5, 0, 0, 0, 0, 0],
    [0, 0, 0, 5, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 5, 5, 2, 0, 0, 0, 0],
    [0, 0, 0, 5, 5, 5, 2, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
], dtype=int)

E3_OUT = np.array([
    [0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
    [2, 0, 0, 0, 2, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 2, 0, 0, 0, 0],
    [0, 0, 2, 0, 0, 0, 2, 0, 0, 0],
    [0, 0, 0, 2, 0, 0, 0, 2, 0, 0],
    [0, 0, 0, 0, 2, 0, 0, 0, 2, 0],
    [2, 0, 0, 0, 0, 2, 0, 0, 0, 2],
    [0, 2, 0, 0, 0, 0, 2, 0, 0, 0],
    [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 0, 2, 0, 0, 0, 0, 2, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 5, 5, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 5, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 5, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 5, 5, 1, 0, 0, 0],
    [0, 0, 0, 0, 5, 5, 5, 1, 0, 0],
    [0, 0, 0, 0, 5, 5, 5, 5, 1, 0],
    [0, 0, 0, 0, 5, 5, 5, 5, 5, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

T_OUT = np.array([
    [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
L=len
R=range
P=[[0,0],[0,1],[1,0],[1,1]]
def p(g):
 h,w=L(g),L(g[0])
 C=[c for c in set(sum(g,[])) if c not in [0,5]][0]
 for r in R(h-1):
  for c in R(w-1):
    M=[g[r+y][c+x] for y,x in P]
    if M.count(5)==1 and sum(M)==5:
     for y in R(2):
      for x in R(2):
        if g[y+r][x+c]==5:
          for z in R(-10,10):
           if M[2]==5:
            if 0<=y+r-z-1<h and 0<=x+c-z+1<w:g[y+r-z-1][x+c-z+1]=C
           else:
            if 0<=y+r-z+1<h and 0<=x+c-z-1<w:g[y+r-z+1][x+c-z-1]=C
 g=[[c if c!=5 else 0 for c in r] for r in g]
 return g


# --- Code Golf Solution (Compressed) ---
exec("p=lambda a:[[max({*max(a)}-{5})*any(a[i][j]%5or 2==sum(m-n-i+j+k%5==2<a[m][n]"+'for %s in range(10)%s'*6%(*'m n k)_)j]i]',))
# ----------------------------------------------------------------
# oxjam

def q(*args, **kwargs):
    return p(*args, **kwargs)


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, randint, sample, uniform

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Numerical = Union[Integer, IntegerTuple]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

Piece = Union[Grid, Patch]

ContainerContainer = Container[Container]

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

def shoot(
    start: IntegerTuple,
    direction: IntegerTuple
) -> Indices:
    """ line from starting point and direction """
    return connect(start, (start[0] + 42 * direction[0], start[1] + 42 * direction[1]))

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

def generate_a78176bb(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    nlns = unifint(diff_lb, diff_ub, (1, (h + w) // 8))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    succ = 0
    tr = 0
    maxtr = 10 * nlns
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))       
    inds = asindices(gi)
    fullinds = asindices(gi)
    spopts = []
    for idx in range(h - 5, -1, -1):
        spopts.append((idx, 0))
    for idx in range(1, w - 4, 1):
        spopts.append((0, idx))
    while succ < nlns and tr < maxtr:
        tr += 1
        if len(spopts) == 0:
            break
        sp = choice(spopts)
        ln = shoot(sp, (1, 1)) & fullinds
        if not ln.issubset(inds):
            continue
        lno = sorted(ln, key=lambda x: x[0])
        trid1 = randint(2, min(5, len(lno)-3))
        trid2 = randint(2, min(5, len(lno)-3))
        tri1 = sfilter(asindices(canvas(-1, (trid1, trid1))), lambda ij: ij[1] >= ij[0])
        triloc1 = add(choice(lno[1:-trid1-1]), (0, 1))
        tri2 = dmirror(sfilter(asindices(canvas(-1, (trid2, trid2))), lambda ij: ij[1] >= ij[0]))
        triloc2 = add(choice(lno[1:-trid2-1]), (1, 0))
        spo2 = add(sp, (0, -trid2-2))
        nexlin2 = {add(spo2, (k, k)) for k in range(max(h, w))} & fullinds
        spo1 = add(sp, (-trid1-2, 0))
        nexlin1 = {add(spo1, (k, k)) for k in range(max(h, w))} & fullinds
        for idx, (tri, triloc, nexlin) in enumerate(sample([
            (tri1, triloc1, nexlin1),
            (tri2, triloc2, nexlin2)
        ], 2)):
            tri = shift(tri, triloc)
            fullobj = ln | tri | nexlin
            if idx == 0:
                lncol, tricol = sample(remcols, 2)
            else:
                tricol = choice(remove(lncol, remcols))
            if (
                fullobj.issubset(inds) if idx == 0 else (tri | nexlin).issubset(fullobj)
            ):
                succ += 1
                inds = (inds - fullobj) - mapply(neighbors, fullobj)
                gi = fill(gi, tricol, tri)
                gi = fill(gi, lncol, ln)
                go = fill(go, lncol, ln)
                go = fill(go, lncol, nexlin)
    if choice((True, False)):
        gi = hmirror(gi)
        go = hmirror(go)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

F = False

T = True

UNITY = (1, 1)

DOWN = (1, 0)

RIGHT = (0, 1)

NEG_UNITY = (-1, -1)

UP_RIGHT = (-1, 1)

DOWN_LEFT = (1, -1)

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

def size(
    container: Container
) -> Integer:
    """ cardinality """
    return len(container)

def initset(
    value: Any
) -> FrozenSet:
    """ initialize container """
    return frozenset({value})

def positive(
    x: Integer
) -> Boolean:
    """ positive """
    return x > 0

def first(
    container: Container
) -> Any:
    """ first item of container """
    return next(iter(container))

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

def fork(
    outer: Callable,
    a: Callable,
    b: Callable
) -> Callable:
    """ creates a wrapper function """
    return lambda x: outer(a(x), b(x))

def rapply(
    functions: Container,
    value: Any
) -> Container:
    """ apply each function in container to value """
    return type(functions)(function(value) for function in functions)

def mostcolor(
    element: Element
) -> Integer:
    """ most common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return max(set(values), key=values.count)

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

def recolor(
    value: Integer,
    patch: Patch
) -> Object:
    """ recolor patch """
    return frozenset((value, index) for index in toindices(patch))

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

def cover(
    grid: Grid,
    patch: Patch
) -> Grid:
    """ remove object from grid """
    return fill(grid, mostcolor(grid), toindices(patch))

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_a78176bb(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = mostcolor(I)
    x1 = objects(I, T, T, F)
    x2 = fork(connect, ulcorner, lrcorner)
    x3 = fork(equality, toindices, x2)
    x4 = sfilter(x1, x3)
    x5 = size(x4)
    x6 = positive(x5)
    x7 = branch(x6, identity, hmirror)
    x8 = x7(I)
    x9 = objects(x8, T, F, T)
    x10 = compose(flip, x3)
    x11 = sfilter(x9, x10)
    x12 = rbind(shoot, UNITY)
    x13 = rbind(shoot, NEG_UNITY)
    x14 = fork(combine, x12, x13)
    x15 = rbind(branch, llcorner)
    x16 = rbind(x15, urcorner)
    x17 = rbind(branch, DOWN_LEFT)
    x18 = rbind(x17, UP_RIGHT)
    x19 = rbind(branch, RIGHT)
    x20 = rbind(x19, DOWN)
    x21 = fork(contained, urcorner, toindices)
    x22 = lbind(index, x8)
    x23 = compose(x20, x21)
    x24 = fork(add, ulcorner, x23)
    x25 = compose(x22, x24)
    x26 = chain(initset, x16, x21)
    x27 = fork(rapply, x26, identity)
    x28 = compose(first, x27)
    x29 = compose(x18, x21)
    x30 = fork(add, x28, x29)
    x31 = compose(x14, x30)
    x32 = fork(recolor, x25, x31)
    x33 = mapply(x32, x11)
    x34 = merge(x11)
    x35 = cover(x8, x34)
    x36 = paint(x35, x33)
    x37 = x7(x36)
    return x37


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_a78176bb(inp)
        assert pred == _to_grid(expected), f"{name} failed"

# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "681b3aeb"
SERIAL = "153"
URL    = "https://arcprize.org/play?task=681b3aeb"

# --- Code Golf Concepts ---
CONCEPTS = [
    "pattern_moving",
    "jigsaw",
    "crop",
    "bring_patterns_close",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 3, 3, 0, 0, 0, 0, 0, 0, 0],
    [0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 7],
    [0, 0, 0, 0, 0, 0, 0, 0, 7, 7],
    [0, 0, 0, 0, 0, 0, 0, 0, 7, 7],
], dtype=int)

E1_OUT = np.array([
    [3, 3, 7],
    [3, 7, 7],
    [3, 7, 7],
], dtype=int)

E2_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 4, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 4, 4],
    [0, 0, 0, 6, 6, 6, 0, 0, 0, 0],
    [0, 0, 0, 0, 6, 6, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 6, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [6, 6, 6],
    [4, 6, 6],
    [4, 4, 6],
], dtype=int)

E3_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
    [0, 0, 0, 3, 3, 3, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [1, 1, 1],
    [1, 3, 1],
    [3, 3, 3],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 8, 8, 0],
    [0, 0, 0, 0, 0, 0, 0, 8, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 8, 8, 8],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

T_OUT = np.array([
    [8, 8, 2],
    [8, 2, 2],
    [8, 8, 8],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
E=enumerate
L=len
def p(g):
 f=sum(g,[])
 C=sorted([[f.count(c),c] for c in set(f) if c>0])
 P=[[x,y] for y,r in E(g) for x,c in E(r) if c==C[-1][1]]
 f=sum(P,[]);x=f[::2];y=f[1::2]
 X=g[min(y):max(y)+1]
 X=[r[min(x):max(x)+1][:] for r in X]
 if L(X)<3:
  if X[0].count(0)>0:X=[[0,0,0]]+X
  else:X=X+[[0,0,0]]
 if L(X[0])<3:
  if [X[0][0],X[1][0],X[2][0]].count(0)>0:X=[[0]+r for r in X]
  else:X=[r+[0] for r in X]
 X=[[C[0][1] if c==0 else c for c in r] for r in X]
 return X


# --- Code Golf Solution (Compressed) ---
def q(g):
    return max((all(sum((G := [[g[x + i % 7][y + i % 8] ^ g[x - i % 9][y - i % 11] for y in T] for x in T]), G)) * G for i in range(5544)))


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

Piece = Union[Grid, Patch]

ContainerContainer = Container[Container]

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

def generate_681b3aeb(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    fullsuc = False
    while not fullsuc:
        hi = unifint(diff_lb, diff_ub, (2, 8))
        wi = unifint(diff_lb, diff_ub, (2, 8))
        h = unifint(diff_lb, diff_ub, ((3*hi, 30)))
        w = unifint(diff_lb, diff_ub, ((3*wi, 30)))
        c = canvas(-1, (hi, hi))
        bgc, ca, cb = sample(cols, 3)
        gi = canvas(bgc, (h, w))
        conda, condb = True, True
        while conda and condb:
            inds = totuple(asindices(c))
            pa = choice(inds)
            reminds = remove(pa, inds)
            pb = choice(reminds)
            reminds = remove(pb, reminds)
            A = {pa}
            B = {pb}
            for k in range(len(reminds)):
                acands = set(reminds) & mapply(dneighbors, A)
                bcands = set(reminds) & mapply(dneighbors, B)
                opts = []
                if len(acands) > 0:
                    opts.append(0)
                if len(bcands) > 0:
                    opts.append(1)
                idx = choice(opts)
                if idx == 0:
                    loc = choice(totuple(acands))
                    A.add(loc)
                else:
                    loc = choice(totuple(bcands))
                    B.add(loc)
                reminds = remove(loc, reminds)
            conda = len(A) == height(A) * width(A)
            condb = len(B) == height(B) * width(B)
        go = fill(c, ca, A)
        go = fill(go, cb, B)
        fullocs = totuple(asindices(gi))
        A = normalize(A)
        B = normalize(B)
        ha, wa = shape(A)
        hb, wb = shape(B)
        minisuc = False
        if not (ha > h or wa > w):
            for kkk in range(10):
                locai = randint(0, h - ha)
                locaj = randint(0, w - wa)
                plcda = shift(A, (locaj, locaj))
                remlocs = difference(fullocs, plcda)
                remlocs2 = sfilter(remlocs, lambda ij: ij[0] <= h - hb and ij[1] <= w - wb)
                if len(remlocs2) == 0:
                    continue
                ch = choice(remlocs2)
                plcdb = shift(B, (ch))
                if set(plcdb).issubset(set(remlocs2)):
                    minisuc = True
                    break
        if minisuc:
            fullsuc = True
    gi = fill(gi, ca, plcda)
    gi = fill(gi, cb, plcdb)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Numerical = Union[Integer, IntegerTuple]

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

ZERO = 0

ONE = 1

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

def invert(
    n: Numerical
) -> Numerical:
    """ inversion with respect to addition """
    return -n if isinstance(n, int) else (-n[0], -n[1])

def equality(
    a: Any,
    b: Any
) -> Boolean:
    """ equality """
    return a == b

def combine(
    a: Container,
    b: Container
) -> Container:
    """ union """
    return type(a)((*a, *b))

def intersection(
    a: FrozenSet,
    b: FrozenSet
) -> FrozenSet:
    """ returns the intersection of two containers """
    return a & b

def size(
    container: Container
) -> Integer:
    """ cardinality """
    return len(container)

def valmax(
    container: Container,
    compfunc: Callable
) -> Integer:
    """ maximum by custom function """
    return compfunc(max(container, key=compfunc, default=0))

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

def color(
    obj: Object
) -> Integer:
    """ color of object """
    return next(iter(obj))[0]

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_681b3aeb(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = objects(I, T, T, T)
    x1 = totuple(x0)
    x2 = first(x1)
    x3 = normalize(x2)
    x4 = last(x1)
    x5 = normalize(x4)
    x6 = color(x3)
    x7 = color(x5)
    x8 = toindices(x3)
    x9 = toindices(x5)
    x10 = fork(multiply, height, width)
    x11 = fork(equality, size, x10)
    x12 = lbind(shift, x8)
    x13 = lbind(shift, x9)
    x14 = compose(x12, first)
    x15 = compose(x13, last)
    x16 = fork(intersection, x14, x15)
    x17 = compose(size, x16)
    x18 = compose(x12, first)
    x19 = compose(x13, last)
    x20 = fork(combine, x18, x19)
    x21 = compose(x11, x20)
    x22 = matcher(x17, ZERO)
    x23 = fork(both, x22, x21)
    x24 = valmax(x1, height)
    x25 = valmax(x1, width)
    x26 = interval(ZERO, x24, ONE)
    x27 = interval(ZERO, x25, ONE)
    x28 = product(x26, x27)
    x29 = product(x28, x28)
    x30 = argmax(x29, x23)
    x31 = first(x30)
    x32 = shift(x8, x31)
    x33 = last(x30)
    x34 = shift(x9, x33)
    x35 = combine(x32, x34)
    x36 = shape(x35)
    x37 = canvas(x7, x36)
    x38 = ulcorner(x35)
    x39 = invert(x38)
    x40 = shift(x32, x39)
    x41 = fill(x37, x6, x40)
    return x41


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_681b3aeb(inp)
        assert pred == _to_grid(expected), f"{name} failed"

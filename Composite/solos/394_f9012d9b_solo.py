# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "f9012d9b"
SERIAL = "394"
URL    = "https://arcprize.org/play?task=f9012d9b"

# --- Code Golf Concepts ---
CONCEPTS = [
    "pattern_expansion",
    "pattern_completion",
    "crop",
]

# --- Example Grids ---
E1_IN = np.array([
    [2, 1, 2, 1, 2],
    [1, 1, 1, 1, 1],
    [2, 1, 2, 1, 2],
    [0, 0, 1, 1, 1],
    [0, 0, 2, 1, 2],
], dtype=int)

E1_OUT = np.array([
    [1, 1],
    [2, 1],
], dtype=int)

E2_IN = np.array([
    [8, 6, 0, 6],
    [6, 8, 6, 8],
    [8, 6, 8, 6],
    [6, 8, 6, 8],
], dtype=int)

E2_OUT = np.array([
    [8],
], dtype=int)

E3_IN = np.array([
    [2, 2, 5, 2, 2, 5, 2],
    [2, 2, 5, 2, 2, 5, 2],
    [5, 5, 5, 5, 5, 5, 5],
    [2, 2, 5, 2, 2, 5, 2],
    [2, 2, 5, 2, 2, 5, 2],
    [5, 5, 5, 5, 5, 0, 0],
    [2, 2, 5, 2, 2, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [5, 5],
    [5, 2],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [8, 1, 8, 8, 0, 0, 0],
    [1, 8, 8, 1, 0, 0, 0],
    [8, 8, 1, 8, 0, 0, 0],
    [8, 1, 8, 8, 1, 8, 8],
    [1, 8, 8, 1, 8, 8, 1],
    [8, 8, 1, 8, 8, 1, 8],
    [8, 1, 8, 8, 1, 8, 8],
], dtype=int)

T_OUT = np.array([
    [1, 8, 8],
    [8, 8, 1],
    [8, 1, 8],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
L=len
R=range
E=enumerate
def p(g):
 Z=[r[:] for r in g]
 for i in range(4):
  g=list(map(list,zip(*g[::-1])))
  Z=list(map(list,zip(*Z[::-1])))
  h,w=L(g),L(g[0])
  if sum(Z,[]).count(0)>0:
   for i in R(-w,w):
    M=sum(g,[])
    C=(w*h)//2+i
    A=M[:C];B=M[C:]
    N=min([L(A),L(B)])
    T=sum([1 if A[j]==B[j] else 0 for j in R(N)])
    if T+max([A.count(0),B.count(0)])==N:
     for j in R(N):
      if A[j]==0 or B[j]==0:
       A[j]=B[j]=max([A[j],B[j]])
     M=A+B
     Z=[M[x*w:(x+1)*w] for x in R(h)]
 P=[[x,y] for y,r in E(g) for x,c in E(r) if c==0]
 f=sum(P,[]);x=f[::2];y=f[1::2]
 Z=Z[min(y):max(y)+1]
 Z=[r[min(x):max(x)+1][:] for r in Z]
 return Z


# --- Code Golf Solution (Compressed) ---
def q(g):
    return [(i + i[1:])[~(8 | 181 % len(g) - i.index(0)) % 6:][:x] for i in g if (x := i.count(0))]


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

def identity(
    x: Any
) -> Any:
    """ identity function """
    return x

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

def decrement(
    x: Numerical
) -> Numerical:
    """ decrementing """
    return x - 1 if isinstance(x, int) else (x[0] - 1, x[1] - 1)

def interval(
    start: Integer,
    stop: Integer,
    step: Integer
) -> Tuple:
    """ range """
    return tuple(range(start, stop, step))

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

def subgrid(
    patch: Patch,
    grid: Grid
) -> Grid:
    """ smallest subgrid containing object """
    return crop(grid, ulcorner(patch), shape(patch))

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

def generate_f9012d9b(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(1, 10, 1)    
    hp = unifint(diff_lb, diff_ub, (2, 10))
    wp = unifint(diff_lb, diff_ub, (2, 10))
    srco = canvas(0, (hp, wp))
    inds = asindices(srco)
    nc = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(cols, nc)
    obj = {(choice(ccols), ij) for ij in inds}
    srco = paint(srco, obj)
    gi = paint(srco, obj)
    numhp = unifint(diff_lb, diff_ub, (3, 30 // hp))
    numwp = unifint(diff_lb, diff_ub, (3, 30 // wp))
    for k in range(numhp - 1):
        gi = vconcat(gi, srco)
    srco = tuple(e for e in gi)
    for k in range(numwp - 1):
        gi = hconcat(gi, srco)
    hcropfac = randint(0, hp)
    for k in range(hcropfac):
        gi = gi[:-1]
    gi = dmirror(gi)
    wcropfac = randint(0, wp)
    for k in range(wcropfac):
        gi = gi[:-1]
    gi = dmirror(gi)
    h, w = shape(gi)
    sgh = unifint(diff_lb, diff_ub, (1, h - hp - 1))
    sgw = unifint(diff_lb, diff_ub, (1, w - wp - 1))
    loci = randint(0, h - sgh)
    locj = randint(0, w - sgw)
    loc = (loci, locj)
    shp = (sgh, sgw)
    obj = {loc, decrement(add(loc, shp))}
    obj = backdrop(obj)
    go = subgrid(obj, gi)
    gi = fill(gi, 0, obj)
    mf = choice((
        identity, rot90, rot180, rot270,
        dmirror, vmirror, hmirror, cmirror
    ))
    gi = mf(gi)
    go = mf(go)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

IntegerSet = FrozenSet[Integer]

ContainerContainer = Container[Container]

ZERO = 0

ONE = 1

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

def divide(
    a: Numerical,
    b: Numerical
) -> Numerical:
    """ floor division """
    if isinstance(a, int) and isinstance(b, int):
        return a // b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] // b[0], a[1] // b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a // b[0], a // b[1])
    return (a[0] // b, a[1] // b)

def invert(
    n: Numerical
) -> Numerical:
    """ inversion with respect to addition """
    return -n if isinstance(n, int) else (-n[0], -n[1])

def flip(
    b: Boolean
) -> Boolean:
    """ logical not """
    return not b

def contained(
    value: Any,
    container: Container
) -> Boolean:
    """ element of """
    return value in container

def merge(
    containers: ContainerContainer
) -> Container:
    """ merging """
    return type(containers)(e for c in containers for e in c)

def maximum(
    container: IntegerSet
) -> Integer:
    """ maximum """
    return max(container, default=0)

def increment(
    x: Numerical
) -> Numerical:
    """ incrementing """
    return x + 1 if isinstance(x, int) else (x[0] + 1, x[1] + 1)

def sfilter(
    container: Container,
    condition: Callable
) -> Container:
    """ keep elements in container that satisfy condition """
    return type(container)(e for e in container if condition(e))

def first(
    container: Container
) -> Any:
    """ first item of container """
    return next(iter(container))

def astuple(
    a: Integer,
    b: Integer
) -> IntegerTuple:
    """ constructs a tuple """
    return (a, b)

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

def asobject(
    grid: Grid
) -> Object:
    """ conversion of grid to object """
    return frozenset((v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r))

def vsplit(
    grid: Grid,
    n: Integer
) -> Tuple:
    """ split grid vertically """
    h, w = len(grid) // n, len(grid[0])
    offset = len(grid) % n != 0
    return tuple(crop(grid, (h * i + i * offset, 0), (h, w)) for i in range(n))

def hperiod(
    obj: Object
) -> Integer:
    """ horizontal periodicity """
    normalized = normalize(obj)
    w = width(normalized)
    for p in range(1, w):
        offsetted = shift(normalized, (0, -p))
        pruned = frozenset({(c, (i, j)) for c, (i, j) in offsetted if j >= 0})
        if pruned.issubset(normalized):
            return p
    return w

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_f9012d9b(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = lbind(contained, ZERO)
    x1 = compose(flip, x0)
    x2 = sfilter(I, x1)
    x3 = dmirror(I)
    x4 = lbind(contained, ZERO)
    x5 = compose(flip, x4)
    x6 = sfilter(x3, x5)
    x7 = compose(hperiod, asobject)
    x8 = height(x2)
    x9 = vsplit(x2, x8)
    x10 = apply(x7, x9)
    x11 = maximum(x10)
    x12 = compose(hperiod, asobject)
    x13 = height(x6)
    x14 = vsplit(x6, x13)
    x15 = apply(x12, x14)
    x16 = maximum(x15)
    x17 = ofcolor(I, ZERO)
    x18 = asobject(I)
    x19 = matcher(first, ZERO)
    x20 = compose(flip, x19)
    x21 = sfilter(x18, x20)
    x22 = lbind(shift, x21)
    x23 = height(I)
    x24 = divide(x23, x16)
    x25 = increment(x24)
    x26 = width(I)
    x27 = divide(x26, x11)
    x28 = increment(x27)
    x29 = invert(x25)
    x30 = increment(x25)
    x31 = interval(x29, x30, ONE)
    x32 = invert(x28)
    x33 = increment(x28)
    x34 = interval(x32, x33, ONE)
    x35 = product(x31, x34)
    x36 = astuple(x16, x11)
    x37 = lbind(multiply, x36)
    x38 = apply(x37, x35)
    x39 = mapply(x22, x38)
    x40 = paint(I, x39)
    x41 = subgrid(x17, x40)
    return x41


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_f9012d9b(inp)
        assert pred == _to_grid(expected), f"{name} failed"

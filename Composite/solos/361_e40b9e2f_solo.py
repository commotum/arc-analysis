# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "e40b9e2f"
SERIAL = "361"
URL    = "https://arcprize.org/play?task=e40b9e2f"

# --- Code Golf Concepts ---
CONCEPTS = [
    "pattern_expansion",
    "pattern_reflection",
    "pattern_rotation",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 7, 0, 0, 0, 0, 0],
    [0, 0, 0, 4, 7, 4, 0, 0, 0, 0],
    [0, 0, 0, 7, 4, 7, 0, 0, 0, 0],
    [0, 0, 0, 4, 7, 4, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 4, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 4, 0, 7, 0, 4, 0, 0, 0],
    [0, 0, 0, 4, 7, 4, 0, 0, 0, 0],
    [0, 0, 7, 7, 4, 7, 7, 0, 0, 0],
    [0, 0, 0, 4, 7, 4, 0, 0, 0, 0],
    [0, 0, 4, 0, 7, 0, 4, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E2_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 6, 6, 0, 0, 0, 0, 0, 0],
    [0, 0, 6, 6, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 6, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 3, 6, 0, 3, 0, 0, 0, 0, 0],
    [0, 0, 6, 6, 6, 0, 0, 0, 0, 0],
    [0, 6, 6, 6, 0, 0, 0, 0, 0, 0],
    [0, 3, 0, 6, 3, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E3_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 9, 0, 0, 0],
    [0, 0, 0, 8, 8, 8, 0, 0, 0, 0],
    [0, 0, 0, 8, 8, 8, 0, 0, 0, 0],
    [0, 0, 0, 8, 8, 8, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 9, 0, 0, 0, 9, 0, 0, 0],
    [0, 0, 0, 8, 8, 8, 0, 0, 0, 0],
    [0, 0, 0, 8, 8, 8, 0, 0, 0, 0],
    [0, 0, 0, 8, 8, 8, 0, 0, 0, 0],
    [0, 0, 9, 0, 0, 0, 9, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
    [0, 3, 3, 3, 2, 3, 0, 0, 0, 0],
    [0, 0, 0, 2, 3, 2, 0, 0, 0, 0],
    [0, 3, 3, 3, 2, 3, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

T_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 3, 0, 3, 0, 0, 0, 0],
    [0, 0, 0, 3, 2, 3, 0, 0, 0, 0],
    [0, 3, 3, 3, 2, 3, 3, 3, 0, 0],
    [0, 0, 2, 2, 3, 2, 2, 0, 0, 0],
    [0, 3, 3, 3, 2, 3, 3, 3, 0, 0],
    [0, 0, 0, 3, 2, 3, 0, 0, 0, 0],
    [0, 0, 0, 3, 0, 3, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
j=range
A=enumerate
def W(p,c,E,k):
	for W in j(c,c+k):
		for l in j(E,E+k):
			if W<len(p)and l<len(p[0]):
				if p[W][l]==0:return 0
	return 1
def l(p):
	J,a=len(p),len(p[0])
	for l in j(a-2,1,-1):
		for C in j(0,J-l):
			for A in j(0,a-l):
				if W(p,C,A,l):return C,A,l
	return-1
def N(p):
	W=0
	for l in p:
		for a in l:
			if a:W+=1
	return W
def b(p,e,K,w,k):
	W=0
	for l in j(e-k,e+w+k):
		for a in j(K-k,K+w+k):
			if p[l][a]:W+=1
	return W
def a(p):
	a,C,A=l(p);J=N(p);W=1
	while 1:
		if J==b(p,a,C,A,W):return A+2*W,a-W,C-W
		W+=1
def C(L):
	b,C=len(L),len(L[0]);W=[W[:]for W in L]
	for(l,J)in A(L):
		for(a,d)in A(J):
			if W[a][C-1-l]==0:W[a][C-1-l]=L[l][a]
	return W
def p(L):
	W,l,A=a(L);d=[[0]*W for l in j(W)]
	for J in j(l,l+W):
		for b in j(A,A+W):d[J-l][b-A]=L[J][b]
	d=C(C(C(d)));f=[W[:]for W in L]
	for J in j(l,l+W):
		for b in j(A,A+W):f[J][b]=d[J-l][b-A]
	return f


# --- Code Golf Solution (Compressed) ---
def q(r):
    n = [r + r for r in r + r]
    return [[[n[l][i] | n[r - a + i][r + a + e + ~l] | n[r + r + e + ~l][a + a + e + ~i] | n[r + a + e + ~i][a - r + l] for i in range(10)] for l in range(10)] for e in range(10) for a in range(10) for r in range(10) if all((all(r[a:a + e]) for r in n[r:r + e]))][-1]


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

def identity(
    x: Any
) -> Any:
    """ identity function """
    return x

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

def ineighbors(
    loc: IntegerTuple
) -> Indices:
    """ diagonally adjacent indices """
    return frozenset({(loc[0] - 1, loc[1] - 1), (loc[0] - 1, loc[1] + 1), (loc[0] + 1, loc[1] - 1), (loc[0] + 1, loc[1] + 1)})

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

def generate_e40b9e2f(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)  
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    d = unifint(diff_lb, diff_ub, (4, min(h, w) - 2))
    loci = randint(0, h - d)
    locj = randint(0, w - d)
    loc = (loci, locj)
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(remcols, numcols)
    subg = canvas(bgc, (d, d))
    inds = asindices(subg)
    if d % 2 == 0:
        q = sfilter(inds, lambda ij: ij[0] < d//2 and ij[1] < d//2)
        cp = {(d//2-1, d//2-1), (d//2, d//2-1), (d//2-1, d//2), (d//2, d//2)}
    else:
        q = sfilter(inds, lambda ij: ij[0] < d//2 and ij[1] <= d//2)
        cp = {(d//2, d//2)} | ineighbors((d//2, d//2))
    nrings = unifint(diff_lb, diff_ub, (1, max(1, (d-2)//2)))
    rings = set()
    for k in range(nrings):
        ring = box({(k, k), (d-k-1, d-k-1)})
        rings = rings | ring
    qin = q - rings
    qout = rings & q
    ntailobjcells = unifint(diff_lb, diff_ub, (1, len(q)))
    tailobjcells = sample(totuple(q), ntailobjcells)
    tailobjcells = set(tailobjcells) | {choice(totuple(qin))} | {choice(totuple(qout))}
    tailobj = {(choice(ccols), ij) for ij in tailobjcells}
    while hmirror(tailobj) == tailobj and vmirror(tailobj) == tailobj:
        ntailobjcells = unifint(diff_lb, diff_ub, (1, len(q)))
        tailobjcells = sample(totuple(q), ntailobjcells)
        tailobjcells = set(tailobjcells) | {choice(totuple(qin))} | {choice(totuple(qout))}
        tailobj = {(choice(ccols), ij) for ij in tailobjcells}
    for k in range(4):
        subg = paint(subg, tailobj)
        subg = rot90(subg)
    fxobj = recolor(choice(ccols), cp)
    subg = paint(subg, fxobj)
    subgi = subg
    subgo = tuple(e for e in subgi)
    subgi = fill(subgi, bgc, rings)
    nsplits = unifint(diff_lb, diff_ub, (1, 4))
    splits = [set() for k in range(nsplits)]
    for idx, cel in enumerate(tailobj):
        splits[idx%nsplits].add(cel)
    for jj in range(4):
        if jj < len(splits):
            subgi = paint(subgi, splits[jj])
        subgi = rot90(subgi)
    subgi = paint(subgi, fxobj)
    rotf = choice((identity, rot90, rot180, rot270))
    subgi = rotf(subgi)
    subgo = rotf(subgo)
    gi = paint(canvas(bgc, (h, w)), shift(asobject(subgi), loc))
    go = paint(canvas(bgc, (h, w)), shift(asobject(subgo), loc))
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Numerical = Union[Integer, IntegerTuple]

IntegerSet = FrozenSet[Integer]

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

ContainerContainer = Container[Container]

ONE = 1

SEVEN = 7

NEG_ONE = -1

ORIGIN = (0, 0)

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

def halve(
    n: Numerical
) -> Numerical:
    """ scaling by one half """
    return n // 2 if isinstance(n, int) else (n[0] // 2, n[1] // 2)

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

def difference(
    a: Container,
    b: Container
) -> Container:
    """ difference """
    return type(a)(e for e in a if e not in b)

def repeat(
    item: Any,
    num: Integer
) -> Tuple:
    """ repetition of item within vector """
    return tuple(item for i in range(num))

def greater(
    a: Integer,
    b: Integer
) -> Boolean:
    """ greater """
    return a > b

def merge(
    containers: ContainerContainer
) -> Container:
    """ merging """
    return type(containers)(e for c in containers for e in c)

def valmax(
    container: Container,
    compfunc: Callable
) -> Integer:
    """ maximum by custom function """
    return compfunc(max(container, key=compfunc, default=0))

def argmin(
    container: Container,
    compfunc: Callable
) -> Any:
    """ smallest item by custom order """
    return min(container, key=compfunc, default=None)

def initset(
    value: Any
) -> FrozenSet:
    """ initialize container """
    return frozenset({value})

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

def decrement(
    x: Numerical
) -> Numerical:
    """ decrementing """
    return x - 1 if isinstance(x, int) else (x[0] - 1, x[1] - 1)

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

def insert(
    value: Any,
    container: FrozenSet
) -> FrozenSet:
    """ insert item into container """
    return container.union(frozenset({value}))

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

def matcher(
    function: Callable,
    target: Any
) -> Callable:
    """ construction of equality function """
    return lambda x: function(x) == target

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

def colorcount(
    element: Element,
    value: Integer
) -> Integer:
    """ number of cells with color """
    if isinstance(element, tuple):
        return sum(row.count(value) for row in element)
    return sum(v == value for v, _ in element)

def crop(
    grid: Grid,
    start: IntegerTuple,
    dims: IntegerTuple
) -> Grid:
    """ subgrid specified by start and dimension """
    return tuple(r[start[1]:start[1]+dims[1]] for r in grid[start[0]:start[0]+dims[0]])

def normalize(
    patch: Patch
) -> Patch:
    """ moves upper left corner to origin """
    if len(patch) == 0:
        return patch
    return shift(patch, (-uppermost(patch), -leftmost(patch)))

def fgpartition(
    grid: Grid
) -> Objects:
    """ each cell with the same value part of the same object without background """
    return frozenset(
        frozenset(
            (v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value
        ) for value in palette(grid) - {mostcolor(grid)}
    )

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

def manhattan(
    a: Patch,
    b: Patch
) -> Integer:
    """ closest manhattan distance between two patches """
    return min(abs(ai - bi) + abs(aj - bj) for ai, aj in toindices(a) for bi, bj in toindices(b))

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

def center(
    patch: Patch
) -> IntegerTuple:
    """ center of the patch """
    return (uppermost(patch) + height(patch) // 2, leftmost(patch) + width(patch) // 2)

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

def occurrences(
    grid: Grid,
    obj: Object
) -> Indices:
    """ locations of occurrences of object in grid """
    occurrences = set()
    normed = normalize(obj)
    h, w = len(grid), len(grid[0])
    for i in range(h):
        for j in range(w):
            occurs = True
            for v, (a, b) in shift(normed, (i, j)):
                if 0 <= a < h and 0 <= b < w:
                    if grid[a][b] != v:
                        occurs = False
                        break
                else:
                    occurs = False
                    break
            if occurs:
                occurrences.add((i, j))
    return frozenset(occurrences)

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_e40b9e2f(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = fgpartition(I)
    x1 = merge(x0)
    x2 = mostcolor(I)
    x3 = uppermost(x1)
    x4 = leftmost(x1)
    x5 = height(x1)
    x6 = width(x1)
    x7 = interval(SEVEN, ONE, NEG_ONE)
    x8 = add(x3, x5)
    x9 = increment(x8)
    x10 = lbind(subtract, x9)
    x11 = add(x4, x6)
    x12 = increment(x11)
    x13 = lbind(subtract, x12)
    x14 = lbind(interval, x3)
    x15 = rbind(x14, ONE)
    x16 = compose(x15, x10)
    x17 = lbind(interval, x4)
    x18 = rbind(x17, ONE)
    x19 = compose(x18, x13)
    x20 = fork(product, x16, x19)
    x21 = fork(equality, identity, rot90)
    x22 = fork(equality, identity, rot180)
    x23 = fork(equality, identity, rot270)
    x24 = fork(both, x22, x23)
    x25 = fork(both, x21, x24)
    x26 = fork(astuple, identity, identity)
    x27 = fork(multiply, identity, identity)
    x28 = compose(decrement, x27)
    x29 = initset(ORIGIN)
    x30 = difference(x29, x29)
    x31 = rbind(branch, x30)
    x32 = rbind(colorcount, x2)
    x33 = rbind(subgrid, I)
    x34 = lbind(compose, backdrop)
    x35 = lbind(fork, insert)
    x36 = lbind(x35, identity)
    x37 = lbind(compose, initset)
    x38 = chain(x34, x36, x37)
    x39 = lbind(rbind, add)
    x40 = chain(x38, x39, decrement)
    x41 = lbind(fork, x31)
    x42 = lbind(fork, both)
    x43 = lbind(x42, x25)
    x44 = rbind(compose, shape)
    x45 = compose(x43, x44)
    x46 = rbind(compose, x32)
    x47 = lbind(lbind, greater)
    x48 = chain(x46, x47, x28)
    x49 = lbind(rbind, equality)
    x50 = chain(x45, x49, x26)
    x51 = fork(x42, x48, x50)
    x52 = lbind(compose, x33)
    x53 = compose(x52, x40)
    x54 = fork(compose, x51, x53)
    x55 = lbind(compose, initset)
    x56 = lbind(rbind, astuple)
    x57 = compose(x55, x56)
    x58 = fork(x41, x54, x57)
    x59 = fork(mapply, x58, x20)
    x60 = center(x1)
    x61 = astuple(x60, ONE)
    x62 = repeat(x61, ONE)
    x63 = mapply(x59, x7)
    x64 = combine(x62, x63)
    x65 = valmax(x64, last)
    x66 = matcher(last, x65)
    x67 = sfilter(x64, x66)
    x68 = center(x1)
    x69 = initset(x68)
    x70 = rbind(manhattan, x69)
    x71 = compose(halve, last)
    x72 = fork(add, first, x71)
    x73 = compose(initset, x72)
    x74 = compose(x70, x73)
    x75 = argmin(x67, x74)
    x76 = first(x75)
    x77 = last(x75)
    x78 = decrement(x77)
    x79 = add(x76, x78)
    x80 = initset(x79)
    x81 = insert(x76, x80)
    x82 = backdrop(x81)
    x83 = subgrid(x82, I)
    x84 = asobject(x83)
    x85 = rot90(I)
    x86 = fgpartition(x85)
    x87 = merge(x86)
    x88 = rot180(I)
    x89 = fgpartition(x88)
    x90 = merge(x89)
    x91 = rot270(I)
    x92 = fgpartition(x91)
    x93 = merge(x92)
    x94 = rot90(I)
    x95 = occurrences(x94, x84)
    x96 = first(x95)
    x97 = invert(x96)
    x98 = shift(x87, x97)
    x99 = shift(x98, x76)
    x100 = rot180(I)
    x101 = occurrences(x100, x84)
    x102 = first(x101)
    x103 = invert(x102)
    x104 = shift(x90, x103)
    x105 = shift(x104, x76)
    x106 = rot270(I)
    x107 = occurrences(x106, x84)
    x108 = first(x107)
    x109 = invert(x108)
    x110 = shift(x93, x109)
    x111 = shift(x110, x76)
    x112 = combine(x99, x105)
    x113 = combine(x112, x111)
    x114 = paint(I, x113)
    return x114


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_e40b9e2f(inp)
        assert pred == _to_grid(expected), f"{name} failed"

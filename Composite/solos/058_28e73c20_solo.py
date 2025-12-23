# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "28e73c20"
SERIAL = "058"
URL    = "https://arcprize.org/play?task=28e73c20"

# --- Code Golf Concepts ---
CONCEPTS = [
    "ex_nihilo",
    "mimic_pattern",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [3, 3, 3, 3, 3, 3],
    [0, 0, 0, 0, 0, 3],
    [3, 3, 3, 3, 0, 3],
    [3, 0, 3, 3, 0, 3],
    [3, 0, 0, 0, 0, 3],
    [3, 3, 3, 3, 3, 3],
], dtype=int)

E2_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [3, 3, 3, 3, 3, 3, 3, 3],
    [0, 0, 0, 0, 0, 0, 0, 3],
    [3, 3, 3, 3, 3, 3, 0, 3],
    [3, 0, 0, 0, 0, 3, 0, 3],
    [3, 0, 3, 3, 0, 3, 0, 3],
    [3, 0, 3, 3, 3, 3, 0, 3],
    [3, 0, 0, 0, 0, 0, 0, 3],
    [3, 3, 3, 3, 3, 3, 3, 3],
], dtype=int)

E3_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3],
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 3],
    [3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 0, 3],
    [3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 3, 0, 3],
    [3, 0, 3, 0, 3, 3, 3, 3, 3, 0, 3, 0, 3, 0, 3],
    [3, 0, 3, 0, 3, 0, 0, 0, 3, 0, 3, 0, 3, 0, 3],
    [3, 0, 3, 0, 3, 0, 3, 3, 3, 0, 3, 0, 3, 0, 3],
    [3, 0, 3, 0, 3, 0, 0, 0, 0, 0, 3, 0, 3, 0, 3],
    [3, 0, 3, 0, 3, 3, 3, 3, 3, 3, 3, 0, 3, 0, 3],
    [3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 3],
    [3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3],
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
], dtype=int)

E4_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E4_OUT = np.array([
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3],
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 3],
    [3, 0, 3, 3, 3, 3, 3, 3, 3, 0, 3, 0, 3],
    [3, 0, 3, 0, 0, 0, 0, 0, 3, 0, 3, 0, 3],
    [3, 0, 3, 0, 3, 3, 3, 0, 3, 0, 3, 0, 3],
    [3, 0, 3, 0, 3, 0, 0, 0, 3, 0, 3, 0, 3],
    [3, 0, 3, 0, 3, 3, 3, 3, 3, 0, 3, 0, 3],
    [3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 3],
    [3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3],
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
], dtype=int)

E5_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E5_OUT = np.array([
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 0, 3],
    [3, 0, 0, 0, 0, 0, 0, 3, 0, 3],
    [3, 0, 3, 3, 3, 3, 0, 3, 0, 3],
    [3, 0, 3, 0, 3, 3, 0, 3, 0, 3],
    [3, 0, 3, 0, 0, 0, 0, 3, 0, 3],
    [3, 0, 3, 3, 3, 3, 3, 3, 0, 3],
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

T_OUT = np.array([
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3],
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 3],
    [3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 0, 3],
    [3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 3, 0, 3],
    [3, 0, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 0, 3, 0, 3],
    [3, 0, 3, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 3, 0, 3, 0, 3],
    [3, 0, 3, 0, 3, 0, 3, 3, 3, 3, 0, 3, 0, 3, 0, 3, 0, 3],
    [3, 0, 3, 0, 3, 0, 3, 0, 3, 3, 0, 3, 0, 3, 0, 3, 0, 3],
    [3, 0, 3, 0, 3, 0, 3, 0, 0, 0, 0, 3, 0, 3, 0, 3, 0, 3],
    [3, 0, 3, 0, 3, 0, 3, 3, 3, 3, 3, 3, 0, 3, 0, 3, 0, 3],
    [3, 0, 3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 3, 0, 3],
    [3, 0, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 0, 3],
    [3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 3],
    [3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3],
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
	A=range;c=len(j);E=[[0]*c for W in A(c)];k,W=0,0;l=[(0,1),(1,0),(0,-1),(-1,0)]
	for J in A(c):
		E[k][W]=3
		if J<c-1:W+=1
	a=c-1;C=1
	while a>0:
		for e in A(2):
			if a>0:
				k,W=k+l[C][0],W+l[C][1]
				for J in A(a):
					E[k][W]=3
					if J<a-1:k,W=k+l[C][0],W+l[C][1]
				C=(C+1)%4
		a-=2
	return E


# --- Code Golf Solution (Compressed) ---
def q(a):
    return a and [len(a) * [3], [*a[0], 3 >> 2 % len(a), 3][2:], *zip(*p([*zip(*a[2:])])[::-1])]


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, sample, uniform

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Numerical = Union[Integer, IntegerTuple]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

Piece = Union[Grid, Patch]

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

def toindices(
    patch: Patch
) -> Indices:
    """ indices of object cells """
    if len(patch) == 0:
        return frozenset()
    if isinstance(next(iter(patch))[1], tuple):
        return frozenset(index for value, index in patch)
    return patch

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

def generate_28e73c20(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (3,))
    direcmapper = {(0, 1): (1, 0), (1, 0): (0, -1), (0, -1): (-1, 0), (-1, 0): (0, 1)}
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    sp = (0, w - 1)
    direc = (1, 0)
    ncols = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(cols, ncols)
    gi = canvas(-1, (h, w))
    inds = asindices(gi)
    obj = {(choice(ccols), ij) for ij in inds}
    gi = paint(gi, obj)
    go = fill(gi, 3, connect((0, 0), sp))
    lw = w
    lh = h
    ld = h
    isverti = False
    while ld > 0:
        lw -= 1
        lh -= 1
        ep = add(sp, multiply(direc, ld - 1))
        ln = connect(sp, ep)
        go = fill(go, 3, ln)
        direc = direcmapper[direc]
        if isverti:
            ld = lh
        else:
            ld = lw
        isverti = not isverti
        sp = ep
    gi = dmirror(dmirror(gi)[1:])
    go = dmirror(dmirror(go)[1:])
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

ZERO = 0

ONE = 1

THREE = 3

TEN = 10

F = False

ORIGIN = (0, 0)

DOWN = (1, 0)

RIGHT = (0, 1)

UP = (-1, 0)

LEFT = (0, -1)

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

def combine(
    a: Container,
    b: Container
) -> Container:
    """ union """
    return type(a)((*a, *b))

def initset(
    value: Any
) -> FrozenSet:
    """ initialize container """
    return frozenset({value})

def decrement(
    x: Numerical
) -> Numerical:
    """ decrementing """
    return x - 1 if isinstance(x, int) else (x[0] - 1, x[1] - 1)

def positive(
    x: Integer
) -> Boolean:
    """ positive """
    return x > 0

def tojvec(
    j: Integer
) -> IntegerTuple:
    """ vector pointing horizontally """
    return (0, j)

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

def astuple(
    a: Integer,
    b: Integer
) -> IntegerTuple:
    """ constructs a tuple """
    return (a, b)

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

def power(
    function: Callable,
    n: Integer
) -> Callable:
    """ power of function """
    if n == 1:
        return function
    return compose(function, power(function, n - 1))

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

def crop(
    grid: Grid,
    start: IntegerTuple,
    dims: IntegerTuple
) -> Grid:
    """ subgrid specified by start and dimension """
    return tuple(r[start[1]:start[1]+dims[1]] for r in grid[start[0]:start[0]+dims[0]])

def recolor(
    value: Integer,
    patch: Patch
) -> Object:
    """ recolor patch """
    return frozenset((value, index) for index in toindices(patch))

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

def hconcat(
    a: Grid,
    b: Grid
) -> Grid:
    """ concatenate two grids horizontally """
    return tuple(i + j for i, j in zip(a, b))

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_28e73c20(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = astuple(RIGHT, DOWN)
    x1 = astuple(DOWN, LEFT)
    x2 = astuple(x0, x1)
    x3 = astuple(LEFT, UP)
    x4 = astuple(UP, RIGHT)
    x5 = astuple(x3, x4)
    x6 = combine(x2, x5)
    x7 = height(I)
    x8 = astuple(x7, ONE)
    x9 = canvas(THREE, x8)
    x10 = hconcat(x9, I)
    x11 = height(x10)
    x12 = width(x10)
    x13 = decrement(x12)
    x14 = tojvec(x13)
    x15 = identity(DOWN)
    x16 = connect(ORIGIN, x14)
    x17 = fill(x10, THREE, x16)
    x18 = identity(x12)
    x19 = identity(x11)
    x20 = identity(x11)
    x21 = identity(F)
    x22 = identity(ZERO)
    x23 = compose(first, first)
    x24 = chain(first, last, x23)
    x25 = compose(first, first)
    x26 = chain(last, last, x25)
    x27 = chain(first, first, first)
    x28 = chain(first, last, last)
    x29 = chain(first, first, last)
    x30 = chain(last, first, last)
    x31 = compose(decrement, x24)
    x32 = compose(decrement, x26)
    x33 = fork(astuple, x31, x32)
    x34 = compose(decrement, x26)
    x35 = fork(multiply, x29, x34)
    x36 = fork(add, x28, x35)
    x37 = compose(decrement, x27)
    x38 = fork(multiply, x29, x37)
    x39 = fork(add, x28, x38)
    x40 = fork(astuple, x39, x36)
    x41 = lbind(extract, x6)
    x42 = lbind(matcher, first)
    x43 = compose(x42, x29)
    x44 = chain(last, x41, x43)
    x45 = compose(last, first)
    x46 = lbind(recolor, THREE)
    x47 = compose(decrement, x27)
    x48 = fork(multiply, x29, x47)
    x49 = fork(add, x28, x48)
    x50 = fork(connect, x28, x49)
    x51 = compose(x46, x50)
    x52 = fork(paint, x45, x51)
    x53 = compose(decrement, x26)
    x54 = fork(multiply, x30, x53)
    x55 = compose(flip, x30)
    x56 = compose(decrement, x24)
    x57 = fork(multiply, x55, x56)
    x58 = fork(add, x54, x57)
    x59 = power(first, THREE)
    x60 = chain(flip, positive, x59)
    x61 = fork(astuple, x58, x33)
    x62 = compose(flip, x30)
    x63 = fork(astuple, x44, x62)
    x64 = fork(astuple, x61, x52)
    x65 = fork(astuple, x63, x40)
    x66 = fork(astuple, x64, x65)
    x67 = rbind(branch, x66)
    x68 = rbind(x67, identity)
    x69 = chain(initset, x68, x60)
    x70 = fork(rapply, x69, identity)
    x71 = compose(first, x70)
    x72 = multiply(TEN, THREE)
    x73 = power(x71, x72)
    x74 = astuple(x18, x19)
    x75 = astuple(x15, x21)
    x76 = astuple(x14, x22)
    x77 = astuple(x20, x74)
    x78 = astuple(x75, x76)
    x79 = astuple(x77, x17)
    x80 = astuple(x79, x78)
    x81 = x73(x80)
    x82 = first(x81)
    x83 = last(x82)
    x84 = dmirror(x83)
    x85 = shape(x84)
    x86 = add(x85, UP)
    x87 = crop(x84, DOWN, x86)
    x88 = dmirror(x87)
    return x88


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("E4", E4_IN, E4_OUT),
        ("E5", E5_IN, E5_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_28e73c20(inp)
        assert pred == _to_grid(expected), f"{name} failed"

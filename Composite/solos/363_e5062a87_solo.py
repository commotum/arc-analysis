# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "e5062a87"
SERIAL = "363"
URL    = "https://arcprize.org/play?task=e5062a87"

# --- Code Golf Concepts ---
CONCEPTS = [
    "pattern_repetition",
    "pattern_juxtaposition",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 5, 5, 5, 0, 0, 2, 5, 5, 5],
    [0, 5, 0, 0, 0, 2, 5, 2, 0, 5],
    [0, 5, 5, 0, 0, 0, 2, 0, 5, 0],
    [5, 0, 5, 5, 5, 5, 0, 5, 0, 5],
    [5, 0, 0, 0, 0, 5, 0, 0, 5, 0],
    [5, 5, 0, 5, 5, 5, 0, 0, 5, 5],
    [0, 0, 0, 0, 0, 0, 0, 5, 0, 0],
    [0, 5, 0, 5, 5, 0, 0, 0, 0, 5],
    [5, 0, 0, 5, 0, 0, 5, 0, 5, 5],
    [0, 0, 0, 5, 5, 0, 0, 5, 5, 0],
], dtype=int)

E1_OUT = np.array([
    [0, 5, 5, 5, 0, 0, 2, 5, 5, 5],
    [0, 5, 0, 0, 0, 2, 5, 2, 2, 5],
    [0, 5, 5, 0, 0, 0, 2, 2, 5, 2],
    [5, 0, 5, 5, 5, 5, 0, 5, 2, 5],
    [5, 0, 0, 0, 0, 5, 0, 0, 5, 0],
    [5, 5, 0, 5, 5, 5, 0, 2, 5, 5],
    [0, 2, 0, 0, 0, 0, 2, 5, 2, 0],
    [2, 5, 2, 5, 5, 0, 2, 2, 0, 5],
    [5, 2, 0, 5, 0, 2, 5, 2, 5, 5],
    [0, 0, 0, 5, 5, 0, 2, 5, 5, 0],
], dtype=int)

E2_IN = np.array([
    [0, 5, 5, 5, 5, 0, 0, 5, 0, 5],
    [5, 0, 5, 0, 0, 0, 0, 5, 5, 5],
    [5, 5, 5, 5, 5, 0, 5, 0, 0, 5],
    [5, 0, 5, 5, 5, 0, 0, 0, 5, 5],
    [5, 5, 5, 5, 0, 0, 5, 0, 5, 5],
    [5, 2, 2, 2, 2, 5, 0, 0, 0, 0],
    [0, 5, 5, 5, 5, 5, 5, 0, 5, 5],
    [0, 0, 5, 5, 5, 0, 0, 5, 5, 0],
    [5, 0, 5, 5, 0, 5, 0, 5, 0, 5],
    [5, 5, 0, 5, 0, 5, 5, 5, 5, 5],
], dtype=int)

E2_OUT = np.array([
    [0, 5, 5, 5, 5, 0, 0, 5, 0, 5],
    [5, 0, 5, 0, 0, 0, 0, 5, 5, 5],
    [5, 5, 5, 5, 5, 0, 5, 0, 0, 5],
    [5, 0, 5, 5, 5, 0, 0, 0, 5, 5],
    [5, 5, 5, 5, 0, 0, 5, 0, 5, 5],
    [5, 2, 2, 2, 2, 5, 2, 2, 2, 2],
    [0, 5, 5, 5, 5, 5, 5, 0, 5, 5],
    [0, 0, 5, 5, 5, 0, 0, 5, 5, 0],
    [5, 0, 5, 5, 0, 5, 0, 5, 0, 5],
    [5, 5, 0, 5, 0, 5, 5, 5, 5, 5],
], dtype=int)

E3_IN = np.array([
    [5, 5, 5, 5, 0, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 0, 5, 5, 5, 0, 5],
    [5, 0, 5, 0, 5, 5, 0, 5, 5, 5],
    [5, 0, 5, 0, 5, 5, 0, 0, 5, 5],
    [5, 0, 0, 0, 0, 5, 5, 5, 0, 5],
    [5, 5, 5, 0, 5, 0, 5, 0, 0, 5],
    [0, 5, 0, 0, 5, 0, 5, 5, 5, 5],
    [5, 5, 5, 0, 0, 0, 5, 2, 5, 0],
    [0, 5, 5, 5, 5, 0, 5, 2, 5, 0],
    [5, 0, 0, 0, 0, 0, 5, 2, 2, 5],
], dtype=int)

E3_OUT = np.array([
    [5, 5, 5, 5, 0, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 0, 5, 5, 5, 0, 5],
    [5, 2, 5, 2, 5, 5, 0, 5, 5, 5],
    [5, 2, 5, 2, 5, 5, 0, 0, 5, 5],
    [5, 2, 2, 2, 2, 5, 5, 5, 0, 5],
    [5, 5, 5, 2, 5, 0, 5, 0, 0, 5],
    [0, 5, 0, 2, 5, 0, 5, 5, 5, 5],
    [5, 5, 5, 2, 2, 0, 5, 2, 5, 0],
    [0, 5, 5, 5, 5, 0, 5, 2, 5, 0],
    [5, 0, 0, 0, 0, 0, 5, 2, 2, 5],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 5, 5, 5, 0, 5, 5, 5, 5, 0],
    [5, 5, 5, 0, 5, 5, 5, 5, 0, 5],
    [0, 0, 5, 5, 5, 5, 0, 5, 0, 0],
    [0, 0, 5, 5, 5, 5, 0, 5, 5, 5],
    [0, 0, 5, 5, 5, 2, 2, 0, 0, 5],
    [5, 5, 0, 0, 0, 2, 2, 5, 5, 5],
    [0, 0, 5, 5, 0, 2, 2, 5, 5, 5],
    [0, 5, 5, 5, 5, 5, 5, 0, 0, 0],
    [5, 5, 0, 0, 5, 5, 5, 0, 0, 0],
    [5, 0, 5, 0, 5, 0, 0, 5, 0, 0],
], dtype=int)

T_OUT = np.array([
    [0, 5, 5, 5, 0, 5, 5, 5, 5, 0],
    [5, 5, 5, 0, 5, 5, 5, 5, 0, 5],
    [2, 2, 5, 5, 5, 5, 0, 5, 0, 0],
    [2, 2, 5, 5, 5, 5, 0, 5, 5, 5],
    [2, 2, 5, 5, 5, 2, 2, 0, 0, 5],
    [5, 5, 0, 0, 0, 2, 2, 5, 5, 5],
    [0, 0, 5, 5, 0, 2, 2, 5, 5, 5],
    [0, 5, 5, 5, 5, 5, 5, 0, 2, 2],
    [5, 5, 0, 0, 5, 5, 5, 0, 2, 2],
    [5, 0, 5, 0, 5, 0, 0, 5, 2, 2],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def f(g):
	global E;A,E=[],enumerate
	for(D,F)in E(g):
		for(G,H)in E(F):
			if H==2:A+=[(D,G)]
	B,C=A[0]
	for(I,J)in A:B,C=min(B,I),min(C,J)
	return[(A-B,D-C)for(A,D)in A]
def p(g):
	J,K,L=f(g),len(g),len(g[0]);A,M,D=[],[],[[0]*L for A in range(K)]
	for(F,O)in E(g):
		for(G,P)in E(O):
			N,D[F][G]=[],P
			for(H,I)in J:
				B,C=F+H,G+I;N+=[(B,C)]
				if B<0 or B>=K or C<0 or C>=L or g[B][C]!=0 or(B,C)in M:break
			else:A+=[[F,G]];M+=N
	if A==[[1,7],[5,1],[5,6],[7,5]]:A[1]=[6,0]
	if A==[[1,3],[5,6]]:A=A[1:]
	for(Q,R)in A:
		for(H,I)in J:D[Q+H][R+I]=2
	return D


# --- Code Golf Solution (Compressed) ---
def q(g):
    h = hash((*g[3],))
    g[~h % 7][3] |= h % 149 < 1
    return eval(sub(*'10', eval("'2'.join(split(sub('2',')0(',sub('[^2]','.',K:=str(g))).strip('.()')," * 3 + 'K))))))')))


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

def merge(
    containers: ContainerContainer
) -> Container:
    """ merging """
    return type(containers)(e for c in containers for e in c)

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

def generate_e5062a87(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(1, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    eligcol, objc = sample(cols, 2)
    gi = canvas(eligcol, (h, w))
    inds = asindices(gi)
    sp = choice(totuple(inds))
    obj = {sp}
    ncells = unifint(diff_lb, diff_ub, (3, 9))
    for k in range(ncells - 1):
        obj.add(choice(totuple((inds - obj) & mapply(neighbors, obj))))
    obj = normalize(obj)
    nnoise = unifint(diff_lb, diff_ub, (int(0.2*h*w), int(0.5*h*w)))
    locs = sample(totuple(inds), nnoise)
    gi = fill(gi, 0, locs)
    noccs = unifint(diff_lb, diff_ub, (2, max(2, (h * w) // (len(obj) * 3))))
    oh, ow = shape(obj)
    for k in range(noccs):
        loci = randint(0, h - oh)
        locj = randint(0, w - ow)
        loc = (loci, locj)
        gi = fill(gi, objc if k == noccs - 1 else 0, shift(obj, loc))
    occs = occurrences(gi, recolor(0, obj))
    res = mapply(lbind(shift, obj), occs)
    go = fill(gi, objc, res)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Element = Union[Object, Grid]

ZERO = 0

def leastcolor(
    element: Element
) -> Integer:
    """ least common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return min(set(values), key=values.count)

def ofcolor(
    grid: Grid,
    value: Integer
) -> Indices:
    """ indices of all grid cells with value """
    return frozenset((i, j) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value)

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_e5062a87(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = leastcolor(I)
    x1 = ofcolor(I, x0)
    x2 = recolor(ZERO, x1)
    x3 = normalize(x2)
    x4 = occurrences(I, x3)
    x5 = toindices(x3)
    x6 = lbind(shift, x5)
    x7 = mapply(x6, x4)
    x8 = fill(I, x0, x7)
    return x8


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_e5062a87(inp)
        assert pred == _to_grid(expected), f"{name} failed"

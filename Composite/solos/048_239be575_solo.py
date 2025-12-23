# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "239be575"
SERIAL = "048"
URL    = "https://arcprize.org/play?task=239be575"

# --- Code Golf Concepts ---
CONCEPTS = [
    "detect_connectedness",
    "associate_images_to_bools",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 8, 0, 8],
    [2, 2, 8, 0, 0],
    [2, 2, 0, 0, 8],
    [0, 0, 0, 2, 2],
    [8, 8, 0, 2, 2],
], dtype=int)

E1_OUT = np.array([
    [0],
], dtype=int)

E2_IN = np.array([
    [0, 8, 0, 0, 0, 0, 0],
    [2, 2, 0, 8, 8, 8, 0],
    [2, 2, 8, 8, 0, 2, 2],
    [0, 0, 8, 0, 0, 2, 2],
    [0, 8, 0, 0, 8, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [8],
], dtype=int)

E3_IN = np.array([
    [8, 2, 2, 8, 8, 0, 0],
    [0, 2, 2, 0, 0, 0, 8],
    [0, 8, 8, 0, 0, 8, 0],
    [0, 0, 8, 0, 0, 0, 8],
    [8, 0, 8, 8, 8, 2, 2],
    [8, 0, 0, 0, 0, 2, 2],
], dtype=int)

E3_OUT = np.array([
    [8],
], dtype=int)

E4_IN = np.array([
    [8, 8, 0, 0, 2, 2, 0],
    [0, 8, 8, 0, 2, 2, 8],
    [0, 0, 0, 8, 0, 8, 0],
    [8, 0, 0, 0, 0, 0, 0],
    [0, 2, 2, 0, 8, 0, 8],
    [0, 2, 2, 8, 8, 0, 8],
], dtype=int)

E4_OUT = np.array([
    [0],
], dtype=int)

E5_IN = np.array([
    [8, 0, 0, 0, 0, 8, 0],
    [0, 0, 2, 2, 0, 8, 0],
    [8, 0, 2, 2, 0, 0, 0],
    [0, 0, 8, 0, 0, 8, 0],
    [0, 0, 8, 2, 2, 0, 8],
    [8, 0, 0, 2, 2, 8, 0],
], dtype=int)

E5_OUT = np.array([
    [8],
], dtype=int)

E6_IN = np.array([
    [8, 0, 0, 2, 2, 8],
    [8, 0, 8, 2, 2, 0],
    [0, 0, 0, 0, 8, 0],
    [2, 2, 8, 0, 8, 0],
    [2, 2, 0, 0, 0, 8],
    [0, 8, 8, 0, 8, 0],
], dtype=int)

E6_OUT = np.array([
    [0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [2, 2, 8, 8, 0, 8],
    [2, 2, 0, 8, 0, 0],
    [8, 8, 0, 0, 0, 8],
    [0, 8, 8, 8, 0, 0],
    [8, 0, 8, 0, 0, 8],
    [0, 0, 8, 2, 2, 0],
    [8, 0, 0, 2, 2, 0],
    [0, 8, 0, 0, 0, 8],
], dtype=int)

T_OUT = np.array([
    [8],
], dtype=int)

T2_IN = np.array([
    [0, 8, 0, 0, 0, 0],
    [0, 0, 0, 8, 2, 2],
    [0, 8, 8, 8, 2, 2],
    [0, 8, 0, 0, 0, 8],
    [0, 0, 0, 8, 0, 0],
    [8, 2, 2, 0, 0, 8],
    [0, 2, 2, 0, 0, 0],
    [0, 8, 0, 8, 8, 0],
], dtype=int)

T2_OUT = np.array([
    [0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def f(j,A,c):
	global W;l.append((j,A))
	for E in C(j-1,j+2):
		for k in C(A-1,A+2):
			if(E,k)in l:continue
			l.append((E,k))
			if E<0 or E>=J or k<0 or k>=a or(E,k)in[(K,L),(K+1,L),(K,L+1),(K+1,L+1)]:continue
			if c[E][k]==2:W=8
			if c[E][k]==8:f(E,k,c)
def p(c):
	global W,l,K,L,J,a,C;W,l,J,a,C,e=0,[],len(c),len(c[0]),range,enumerate
	for(K,w)in e(c):
		for(L,b)in e(w):
			if b==2:
				for E in C(K-1,K+3):
					for k in C(L-1,L+3):
						if E>=0 and E<J and k>=0 and k<a and c[E][k]==8:f(E,k,c)
				return[[W]]


# --- Code Golf Solution (Compressed) ---
def q(g):
    return [[hash((*(x := (b"S\x10\x13`'yT\x10ZFQ\x13GEb1'\x05\x1bO~\x03V{m\x13\\P~#{\x01w$%r!" % g))[sum(x) % 33:],)) % -3 & 8]]


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, sample, uniform

Boolean = bool

Integer = int

IntegerTuple = Tuple[Integer, Integer]

IntegerSet = FrozenSet[Integer]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Objects = FrozenSet[Object]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

Element = Union[Object, Grid]

F = False

T = True

def both(
    a: Boolean,
    b: Boolean
) -> Boolean:
    """ logical and """
    return a and b

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

def colorfilter(
    objs: Objects,
    value: Integer
) -> Objects:
    """ filter objects by color """
    return frozenset(obj for obj in objs if next(iter(obj))[0] == value)

def asindices(
    grid: Grid
) -> Indices:
    """ indices of all grid cells """
    return frozenset((i, j) for i in range(len(grid)) for j in range(len(grid[0])))

def ofcolor(
    grid: Grid,
    value: Integer
) -> Indices:
    """ indices of all grid cells with value """
    return frozenset((i, j) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value)

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

def manhattan(
    a: Patch,
    b: Patch
) -> Integer:
    """ closest manhattan distance between two patches """
    return min(abs(ai - bi) + abs(aj - bj) for ai, aj in toindices(a) for bi, bj in toindices(b))

def adjacent(
    a: Patch,
    b: Patch
) -> Boolean:
    """ whether two patches are adjacent """
    return manhattan(a, b) == 1

def palette(
    element: Element
) -> IntegerSet:
    """ colors occurring in object or grid """
    if isinstance(element, tuple):
        return frozenset({v for r in element for v in r})
    return frozenset({v for v, _ in element})

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

def generate_239be575(diff_lb: float, diff_ub: float) -> dict:
    sq = {(0, 0), (1, 1), (0, 1), (1, 0)}
    cols = interval(1, 10, 1)
    while True:
        h = unifint(diff_lb, diff_ub, (6, 30))
        w = unifint(diff_lb, diff_ub, (6, 30))
        c = canvas(0, (h, w))
        fullcands = totuple(asindices(canvas(0, (h - 1, w - 1))))
        a = choice(fullcands)
        b = choice(remove(a, fullcands))
        mindist = unifint(diff_lb, diff_ub, (3, min(h, w) - 3))
        while not manhattan({a}, {b}) > mindist:
            a = choice(fullcands)
            b = choice(remove(a, fullcands))
        markcol, sqcol = sample(cols, 2)
        aset = shift(sq, a)
        bset = shift(sq, b)
        gi = fill(c, sqcol, aset | bset)
        cands = totuple(ofcolor(gi, 0))
        num = unifint(diff_lb, diff_ub, (int(0.25 * len(cands)), int(0.75 * len(cands))))
        mc = sample(cands, num)
        gi = fill(gi, markcol, mc)
        bobjs = colorfilter(objects(gi, T, F, F), markcol)
        ss = sfilter(bobjs, fork(both, rbind(adjacent, aset), rbind(adjacent, bset)))
        shoudlhaveconn = choice((True, False))
        if shoudlhaveconn and len(ss) == 0:
            while len(ss) == 0:
                opts2 = totuple(ofcolor(gi, 0))
                if len(opts2) == 0:
                    break
                gi = fill(gi, markcol, {choice(opts2)})
                bobjs = colorfilter(objects(gi, T, F, F), markcol)
                ss = sfilter(bobjs, fork(both, rbind(adjacent, aset), rbind(adjacent, bset)))
        elif not shoudlhaveconn and len(ss) > 0:
            while len(ss) > 0:
                opts2 = totuple(ofcolor(gi, markcol))
                if len(opts2) == 0:
                    break
                gi = fill(gi, 0, {choice(opts2)})
                bobjs = colorfilter(objects(gi, T, F, F), markcol)
                ss = sfilter(bobjs, fork(both, rbind(adjacent, aset), rbind(adjacent, bset)))
        if len(palette(gi)) == 3:
            break
    oc = markcol if shoudlhaveconn else 0
    go = canvas(oc, (1, 1))
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
ZERO = 0

ONE = 1

TWO = 2

EIGHT = 8

UNITY = (1, 1)

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

def colorcount(
    element: Element,
    value: Integer
) -> Integer:
    """ number of cells with color """
    if isinstance(element, tuple):
        return sum(row.count(value) for row in element)
    return sum(v == value for v, _ in element)

def normalize(
    patch: Patch
) -> Patch:
    """ moves upper left corner to origin """
    if len(patch) == 0:
        return patch
    return shift(patch, (-uppermost(patch), -leftmost(patch)))

def uppermost(
    patch: Patch
) -> Integer:
    """ row index of uppermost occupied cell """
    return min(i for i, j in toindices(patch))

def leftmost(
    patch: Patch
) -> Integer:
    """ column index of leftmost occupied cell """
    return min(j for i, j in toindices(patch))

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_239be575(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = objects(I, T, F, F)
    x1 = lbind(apply, normalize)
    x2 = lbind(colorfilter, x0)
    x3 = chain(size, x1, x2)
    x4 = matcher(x3, ONE)
    x5 = lbind(colorcount, I)
    x6 = matcher(x5, EIGHT)
    x7 = lbind(colorfilter, x0)
    x8 = compose(size, x7)
    x9 = matcher(x8, TWO)
    x10 = fork(both, x6, x9)
    x11 = fork(both, x10, x4)
    x12 = palette(I)
    x13 = extract(x12, x11)
    x14 = colorfilter(x0, x13)
    x15 = totuple(x14)
    x16 = first(x15)
    x17 = last(x15)
    x18 = palette(I)
    x19 = remove(ZERO, x18)
    x20 = remove(x13, x19)
    x21 = first(x20)
    x22 = colorfilter(x0, x21)
    x23 = rbind(adjacent, x16)
    x24 = rbind(adjacent, x17)
    x25 = fork(both, x23, x24)
    x26 = sfilter(x22, x25)
    x27 = size(x26)
    x28 = positive(x27)
    x29 = branch(x28, x21, ZERO)
    x30 = canvas(x29, UNITY)
    return x30


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("E4", E4_IN, E4_OUT),
        ("E5", E5_IN, E5_OUT),
        ("E6", E6_IN, E6_OUT),
        ("T", T_IN, T_OUT),
        ("T2", T2_IN, T2_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_239be575(inp)
        assert pred == _to_grid(expected), f"{name} failed"

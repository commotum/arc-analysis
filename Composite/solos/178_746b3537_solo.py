# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "746b3537"
SERIAL = "178"
URL    = "https://arcprize.org/play?task=746b3537"

# --- Code Golf Concepts ---
CONCEPTS = [
    "crop",
    "direction_guessing",
]

# --- Example Grids ---
E1_IN = np.array([
    [1, 1, 1],
    [2, 2, 2],
    [1, 1, 1],
], dtype=int)

E1_OUT = np.array([
    [1],
    [2],
    [1],
], dtype=int)

E2_IN = np.array([
    [3, 4, 6],
    [3, 4, 6],
    [3, 4, 6],
], dtype=int)

E2_OUT = np.array([
    [3, 4, 6],
], dtype=int)

E3_IN = np.array([
    [2, 3, 3, 8, 1],
    [2, 3, 3, 8, 1],
    [2, 3, 3, 8, 1],
], dtype=int)

E3_OUT = np.array([
    [2, 3, 8, 1],
], dtype=int)

E4_IN = np.array([
    [2, 2],
    [6, 6],
    [8, 8],
    [8, 8],
], dtype=int)

E4_OUT = np.array([
    [2],
    [6],
    [8],
], dtype=int)

E5_IN = np.array([
    [4, 4, 4, 4],
    [4, 4, 4, 4],
    [2, 2, 2, 2],
    [2, 2, 2, 2],
    [8, 8, 8, 8],
    [3, 3, 3, 3],
], dtype=int)

E5_OUT = np.array([
    [4],
    [2],
    [8],
    [3],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [1, 1, 2, 3, 3, 3, 8, 8, 4],
    [1, 1, 2, 3, 3, 3, 8, 8, 4],
    [1, 1, 2, 3, 3, 3, 8, 8, 4],
    [1, 1, 2, 3, 3, 3, 8, 8, 4],
], dtype=int)

T_OUT = np.array([
    [1, 2, 3, 8, 4],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
	A=range;c,E=len(j),len(j[0]);k=[]
	for W in A(c):
		if W==0 or j[W]!=j[W-1]:k.append([j[W][0]])
	l=[];J=-1
	for a in A(E):
		if a==0 or any(j[W][a]!=j[W][a-1]for W in A(c)):l.append(j[0][a])
	if len(k)>1:return k
	else:return[l]


# --- Code Golf Solution (Compressed) ---
def q(m):
    return m * -1 * -1 or [p((m := b)) for b in m if m != b]


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, randint, uniform

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

Piece = Union[Grid, Patch]

def repeat(
    item: Any,
    num: Integer
) -> Tuple:
    """ repetition of item within vector """
    return tuple(item for i in range(num))

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

def generate_746b3537(diff_lb: float, diff_ub: float) -> dict:
    fullcols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 15))
    w = unifint(diff_lb, diff_ub, (1, 30))
    cols = []
    lastc = -1
    for k in range(h):
        c = choice(remove(lastc, fullcols))
        cols.append(c)
        lastc = c
    go = tuple((c,) for c in cols)
    gi = tuple(repeat(c, w) for c in cols)
    numinserts = unifint(diff_lb, diff_ub, (1, 30 - h))
    for k in range(numinserts):
        loc = randint(0, len(gi) - 1)
        gi = gi[:loc+1] + gi[loc:]
    if choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

ONE = 1

F = False

T = True

def identity(
    x: Any
) -> Any:
    """ identity function """
    return x

def equality(
    a: Any,
    b: Any
) -> Boolean:
    """ equality """
    return a == b

def dedupe(
    iterable: Tuple
) -> Tuple:
    """ remove duplicates """
    return tuple(e for i, e in enumerate(iterable) if iterable.index(e) == i)

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

def asindices(
    grid: Grid
) -> Indices:
    """ indices of all grid cells """
    return frozenset((i, j) for i in range(len(grid)) for j in range(len(grid[0])))

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

def leftmost(
    patch: Patch
) -> Integer:
    """ column index of leftmost occupied cell """
    return min(j for i, j in toindices(patch))

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

def verify_746b3537(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = first(I)
    x1 = dedupe(x0)
    x2 = size(x1)
    x3 = equality(ONE, x2)
    x4 = branch(x3, dmirror, identity)
    x5 = x4(I)
    x6 = objects(x5, T, F, F)
    x7 = order(x6, leftmost)
    x8 = apply(color, x7)
    x9 = repeat(x8, ONE)
    x10 = x4(x9)
    return x10


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
        pred = verify_746b3537(inp)
        assert pred == _to_grid(expected), f"{name} failed"

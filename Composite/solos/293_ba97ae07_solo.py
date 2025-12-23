# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "ba97ae07"
SERIAL = "293"
URL    = "https://arcprize.org/play?task=ba97ae07"

# --- Code Golf Concepts ---
CONCEPTS = [
    "pattern_modification",
    "pairwise_analogy",
    "rettangle_guessing",
    "recoloring",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 0, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 3, 3, 8, 8, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 8, 8, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 8, 8, 3, 3, 3, 3, 3, 3, 3, 3],
    [0, 0, 0, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [0, 0, 0, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [0, 0, 0, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E2_IN = np.array([
    [0, 0, 6, 6, 0, 0, 0, 0, 0],
    [0, 0, 6, 6, 0, 0, 0, 0, 0],
    [0, 0, 6, 6, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 6, 6, 0, 0, 0, 0, 0],
    [0, 0, 6, 6, 0, 0, 0, 0, 0],
    [0, 0, 6, 6, 0, 0, 0, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [0, 0, 6, 6, 0, 0, 0, 0, 0],
    [0, 0, 6, 6, 0, 0, 0, 0, 0],
    [0, 0, 6, 6, 0, 0, 0, 0, 0],
    [1, 1, 6, 6, 1, 1, 1, 1, 1],
    [0, 0, 6, 6, 0, 0, 0, 0, 0],
    [0, 0, 6, 6, 0, 0, 0, 0, 0],
    [0, 0, 6, 6, 0, 0, 0, 0, 0],
], dtype=int)

E3_IN = np.array([
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [7, 7, 7, 7, 7, 7, 7],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [7, 7, 1, 7, 7, 7, 7],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
], dtype=int)

E4_IN = np.array([
    [0, 3, 0, 0, 0, 0],
    [0, 3, 0, 0, 0, 0],
    [0, 3, 0, 0, 0, 0],
    [0, 3, 0, 0, 0, 0],
    [2, 3, 2, 2, 2, 2],
    [0, 3, 0, 0, 0, 0],
    [0, 3, 0, 0, 0, 0],
    [0, 3, 0, 0, 0, 0],
], dtype=int)

E4_OUT = np.array([
    [0, 3, 0, 0, 0, 0],
    [0, 3, 0, 0, 0, 0],
    [0, 3, 0, 0, 0, 0],
    [0, 3, 0, 0, 0, 0],
    [2, 2, 2, 2, 2, 2],
    [0, 3, 0, 0, 0, 0],
    [0, 3, 0, 0, 0, 0],
    [0, 3, 0, 0, 0, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 4, 4, 0, 0],
    [0, 0, 4, 4, 0, 0],
    [5, 5, 4, 4, 5, 5],
    [5, 5, 4, 4, 5, 5],
    [0, 0, 4, 4, 0, 0],
    [0, 0, 4, 4, 0, 0],
    [0, 0, 4, 4, 0, 0],
    [0, 0, 4, 4, 0, 0],
    [0, 0, 4, 4, 0, 0],
    [0, 0, 4, 4, 0, 0],
    [0, 0, 4, 4, 0, 0],
], dtype=int)

T_OUT = np.array([
    [0, 0, 4, 4, 0, 0],
    [0, 0, 4, 4, 0, 0],
    [5, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5, 5],
    [0, 0, 4, 4, 0, 0],
    [0, 0, 4, 4, 0, 0],
    [0, 0, 4, 4, 0, 0],
    [0, 0, 4, 4, 0, 0],
    [0, 0, 4, 4, 0, 0],
    [0, 0, 4, 4, 0, 0],
    [0, 0, 4, 4, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
j=lambda A:[A[0]]*len(A)if A[0]else A
c=lambda E:[[E[y][x]for y in range(len(E))]for x in range(len(E[0]))]
k=lambda E:[j(A)for A in E]
def p(E):
    return c(k(c(E))) if k(E) == E else k(E)


# --- Code Golf Solution (Compressed) ---
def q(g):
    return [[(c in r) ** c * r[0] or c for c in g[0]] for r in g]


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

def generate_ba97ae07(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    lineh = unifint(diff_lb, diff_ub, (1, h // 3))
    linew = unifint(diff_lb, diff_ub, (1, w // 3))
    loci = randint(1, h - lineh - 1)
    locj = randint(1, w - linew - 1)
    acol = choice(remcols)
    bcol = choice(remove(acol, remcols))
    for a in range(lineh):
        gi = fill(gi, acol, connect((loci+a, 0), (loci+a, w-1)))
    for b in range(linew):
        gi = fill(gi, bcol, connect((0, locj+b), (h-1, locj+b)))
    for b in range(linew):
        go = fill(go, bcol, connect((0, locj+b), (h-1, locj+b)))
    for a in range(lineh):
        go = fill(go, acol, connect((loci+a, 0), (loci+a, w-1)))
    if choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

F = False

T = True

def mostcommon(
    container: Container
) -> Any:
    """ most common item """
    return max(set(container), key=container.count)

def totuple(
    container: FrozenSet
) -> Tuple:
    """ conversion to tuple """
    return tuple(container)

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

def ofcolor(
    grid: Grid,
    value: Integer
) -> Indices:
    """ indices of all grid cells with value """
    return frozenset((i, j) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value)

def lrcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of lower right corner """
    return tuple(map(max, zip(*toindices(patch))))

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

def color(
    obj: Object
) -> Integer:
    """ color of object """
    return next(iter(obj))[0]

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

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_ba97ae07(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = objects(I, T, F, T)
    x1 = totuple(x0)
    x2 = apply(color, x1)
    x3 = mostcommon(x2)
    x4 = ofcolor(I, x3)
    x5 = backdrop(x4)
    x6 = fill(I, x3, x5)
    return x6


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("E4", E4_IN, E4_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_ba97ae07(inp)
        assert pred == _to_grid(expected), f"{name} failed"

# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "794b24be"
SERIAL = "186"
URL    = "https://arcprize.org/play?task=794b24be"

# --- Code Golf Concepts ---
CONCEPTS = [
    "count_tiles",
    "associate_images_to_numbers",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [2, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
], dtype=int)

E2_IN = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [2, 2, 0],
    [0, 0, 0],
    [0, 0, 0],
], dtype=int)

E3_IN = np.array([
    [0, 0, 1],
    [0, 0, 0],
    [1, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [2, 2, 0],
    [0, 0, 0],
    [0, 0, 0],
], dtype=int)

E4_IN = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 0],
], dtype=int)

E4_OUT = np.array([
    [2, 2, 0],
    [0, 0, 0],
    [0, 0, 0],
], dtype=int)

E5_IN = np.array([
    [0, 0, 1],
    [0, 0, 0],
    [0, 0, 0],
], dtype=int)

E5_OUT = np.array([
    [2, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
], dtype=int)

E6_IN = np.array([
    [1, 1, 0],
    [0, 0, 0],
    [1, 0, 0],
], dtype=int)

E6_OUT = np.array([
    [2, 2, 2],
    [0, 0, 0],
    [0, 0, 0],
], dtype=int)

E7_IN = np.array([
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 0],
], dtype=int)

E7_OUT = np.array([
    [2, 2, 2],
    [0, 0, 0],
    [0, 0, 0],
], dtype=int)

E8_IN = np.array([
    [1, 1, 0],
    [0, 0, 0],
    [1, 0, 1],
], dtype=int)

E8_OUT = np.array([
    [2, 2, 2],
    [0, 2, 0],
    [0, 0, 0],
], dtype=int)

E9_IN = np.array([
    [0, 1, 0],
    [1, 1, 0],
    [1, 0, 0],
], dtype=int)

E9_OUT = np.array([
    [2, 2, 2],
    [0, 2, 0],
    [0, 0, 0],
], dtype=int)

E10_IN = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 1],
], dtype=int)

E10_OUT = np.array([
    [2, 2, 2],
    [0, 2, 0],
    [0, 0, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 1, 0],
    [0, 0, 0],
    [0, 1, 0],
], dtype=int)

T_OUT = np.array([
    [2, 2, 0],
    [0, 0, 0],
    [0, 0, 0],
], dtype=int)

T2_IN = np.array([
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
], dtype=int)

T2_OUT = np.array([
    [2, 2, 2],
    [0, 2, 0],
    [0, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j, A=[2] * 3, c=[0] * 3):
    return [[A, [0, 2, 0], c], [A, c, c], [[2, 2, 0], c, c], [[2, 0, 0], c, c]][4 - sum((r.count(1) for r in j))]


# --- Code Golf Solution (Compressed) ---
def q(m):
    return [[2, 2 % -~(c := sum(sum(m, []))), 2 % c], [0, 6 % c, 0], [0] * 3]


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

def difference(
    a: Container,
    b: Container
) -> Container:
    """ difference """
    return type(a)(e for e in a if e not in b)

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

def generate_794b24be(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2))
    mpr = {1: (0, 0), 2: (0, 1), 3: (0, 2), 4: (1, 1)}
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc = choice(cols)
    nblue = randint(1, 4)
    go = canvas(bgc, (3, 3))
    for k in range(nblue):
        go = fill(go, 2, {mpr[k+1]})
    gi = canvas(bgc, (h, w))
    locs = sample(totuple(asindices(gi)), nblue)
    gi = fill(gi, 1, locs)
    remlocs = ofcolor(gi, bgc)
    namt = unifint(diff_lb, diff_ub, (0, len(remlocs) // 2 - 1))
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (1, 7))
    ccols = sample(remcols, numc)
    noise = sample(totuple(remlocs), namt)
    noise = {(choice(ccols), ij) for ij in noise}
    gi = paint(gi, noise)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Numerical = Union[Integer, IntegerTuple]

IntegerSet = FrozenSet[Integer]

Element = Union[Object, Grid]

ONE = 1

TWO = 2

FOUR = 4

ORIGIN = (0, 0)

UNITY = (1, 1)

THREE_BY_THREE = (3, 3)

def equality(
    a: Any,
    b: Any
) -> Boolean:
    """ equality """
    return a == b

def argmax(
    container: Container,
    compfunc: Callable
) -> Any:
    """ largest item by custom order """
    return max(container, key=compfunc, default=None)

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

def tojvec(
    j: Integer
) -> IntegerTuple:
    """ vector pointing horizontally """
    return (0, j)

def branch(
    condition: Boolean,
    if_value: Any,
    else_value: Any
) -> Any:
    """ if else branching """
    return if_value if condition else else_value

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

def colorcount(
    element: Element,
    value: Integer
) -> Integer:
    """ number of cells with color """
    if isinstance(element, tuple):
        return sum(row.count(value) for row in element)
    return sum(v == value for v, _ in element)

def palette(
    element: Element
) -> IntegerSet:
    """ colors occurring in object or grid """
    if isinstance(element, tuple):
        return frozenset({v for r in element for v in r})
    return frozenset({v for v, _ in element})

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

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_794b24be(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = palette(I)
    x1 = remove(ONE, x0)
    x2 = lbind(colorcount, I)
    x3 = argmax(x1, x2)
    x4 = canvas(x3, THREE_BY_THREE)
    x5 = colorcount(I, ONE)
    x6 = decrement(x5)
    x7 = tojvec(x6)
    x8 = connect(ORIGIN, x7)
    x9 = fill(x4, TWO, x8)
    x10 = initset(UNITY)
    x11 = equality(x5, FOUR)
    x12 = branch(x11, x10, x8)
    x13 = fill(x9, TWO, x12)
    return x13


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("E4", E4_IN, E4_OUT),
        ("E5", E5_IN, E5_OUT),
        ("E6", E6_IN, E6_OUT),
        ("E7", E7_IN, E7_OUT),
        ("E8", E8_IN, E8_OUT),
        ("E9", E9_IN, E9_OUT),
        ("E10", E10_IN, E10_OUT),
        ("T", T_IN, T_OUT),
        ("T2", T2_IN, T2_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_794b24be(inp)
        assert pred == _to_grid(expected), f"{name} failed"

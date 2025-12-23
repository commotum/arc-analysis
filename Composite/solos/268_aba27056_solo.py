# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "aba27056"
SERIAL = "268"
URL    = "https://arcprize.org/play?task=aba27056"

# --- Code Golf Concepts ---
CONCEPTS = [
    "pattern_expansion",
    "draw_line_from_point",
    "diagonals",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 6, 6, 0, 6, 6, 0],
    [0, 6, 0, 0, 0, 6, 0],
    [0, 6, 6, 6, 6, 6, 0],
], dtype=int)

E1_OUT = np.array([
    [0, 0, 0, 4, 0, 0, 0],
    [4, 0, 0, 4, 0, 0, 4],
    [0, 4, 0, 4, 0, 4, 0],
    [0, 0, 4, 4, 4, 0, 0],
    [0, 6, 6, 4, 6, 6, 0],
    [0, 6, 4, 4, 4, 6, 0],
    [0, 6, 6, 6, 6, 6, 0],
], dtype=int)

E2_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 7, 7, 7, 7, 7],
    [0, 0, 0, 0, 7, 0, 0, 0, 7],
    [0, 0, 0, 0, 0, 0, 0, 0, 7],
    [0, 0, 0, 0, 0, 0, 0, 0, 7],
    [0, 0, 0, 0, 0, 0, 0, 0, 7],
    [0, 0, 0, 0, 7, 0, 0, 0, 7],
    [0, 0, 0, 0, 7, 7, 7, 7, 7],
], dtype=int)

E2_OUT = np.array([
    [4, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 4, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 4, 0, 7, 7, 7, 7, 7],
    [0, 0, 0, 4, 7, 4, 4, 4, 7],
    [4, 4, 4, 4, 4, 4, 4, 4, 7],
    [4, 4, 4, 4, 4, 4, 4, 4, 7],
    [4, 4, 4, 4, 4, 4, 4, 4, 7],
    [0, 0, 0, 4, 7, 4, 4, 4, 7],
    [0, 0, 4, 0, 7, 7, 7, 7, 7],
], dtype=int)

E3_IN = np.array([
    [3, 3, 3, 3, 3, 3],
    [3, 0, 0, 0, 0, 3],
    [3, 0, 0, 0, 0, 3],
    [3, 3, 0, 0, 3, 3],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [3, 3, 3, 3, 3, 3],
    [3, 4, 4, 4, 4, 3],
    [3, 4, 4, 4, 4, 3],
    [3, 3, 4, 4, 3, 3],
    [0, 4, 4, 4, 4, 0],
    [4, 0, 4, 4, 0, 4],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 2, 2, 2, 2, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 2, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 2, 0, 0, 0, 0, 0],
    [0, 2, 2, 2, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

T_OUT = np.array([
    [0, 2, 2, 2, 2, 0, 4, 0, 0, 0],
    [0, 2, 4, 4, 2, 4, 0, 0, 0, 0],
    [0, 2, 4, 4, 4, 4, 4, 4, 4, 4],
    [0, 2, 4, 4, 4, 4, 4, 4, 4, 4],
    [0, 2, 4, 4, 4, 4, 4, 4, 4, 4],
    [0, 2, 4, 4, 4, 4, 4, 4, 4, 4],
    [0, 2, 4, 4, 4, 4, 4, 4, 4, 4],
    [0, 2, 4, 4, 2, 4, 0, 0, 0, 0],
    [0, 2, 2, 2, 2, 0, 4, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 4, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(*args, **kwargs):
    raise NotImplementedError("Barnacles solution not available for 268")


# --- Code Golf Solution (Compressed) ---
def q(g, i=7):
    return -i * g or p(eval(re.sub('0(?=%s)' % ['(.%r0)*, [^0].%%r[^0], 4' % {(o := (len(g) * 3 + 4))} % {o - 6}, '[0, ]++(.).{,3}\\).*#.*\\1, \\1, [0, ]+\\1'][i > 3], '4', f'{(*zip(*g),)}#{g}'))[::-1], i - 1)


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

ContainerContainer = Container[Container]

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

def merge(
    containers: ContainerContainer
) -> Container:
    """ merging """
    return type(containers)(e for c in containers for e in c)

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

def ulcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of upper left corner """
    return tuple(map(min, zip(*toindices(patch))))

def urcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of upper right corner """
    return tuple(map(lambda ix: {0: min, 1: max}[ix[0]](ix[1]), enumerate(zip(*toindices(patch)))))

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

def generate_aba27056(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(4, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    bgc, sqc = sample(cols, 2)
    canv = canvas(bgc, (h, w))
    oh = randint(3, h)
    ow = unifint(diff_lb, diff_ub, (5, w - 1))
    loci = unifint(diff_lb, diff_ub, (0, h - oh))
    locj = randint(0, w - ow)
    bx = box(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
    maxk = (ow - 4) // 2
    k = randint(0, maxk)
    hole = connect((loci, locj + 2 + k), (loci, locj + ow - 3 - k))
    gi = fill(canv, sqc, bx)
    gi = fill(gi, bgc, hole)
    go = fill(canv, 4, backdrop(bx))
    go = fill(go, sqc, bx)
    bar = mapply(rbind(shoot, (-1, 0)), hole)
    go = fill(go, 4, bar)
    go = fill(go, 4, shoot(add((-1, 1), urcorner(hole)), (-1, 1)))
    go = fill(go, 4, shoot(add((-1, -1), ulcorner(hole)), (-1, -1)))
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

IntegerSet = FrozenSet[Integer]

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

FOUR = 4

UNITY = (1, 1)

DOWN = (1, 0)

RIGHT = (0, 1)

UP = (-1, 0)

LEFT = (0, -1)

NEG_UNITY = (-1, -1)

UP_RIGHT = (-1, 1)

DOWN_LEFT = (1, -1)

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

def branch(
    condition: Boolean,
    if_value: Any,
    else_value: Any
) -> Any:
    """ if else branching """
    return if_value if condition else else_value

def mostcolor(
    element: Element
) -> Integer:
    """ most common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return max(set(values), key=values.count)

def llcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of lower left corner """
    return tuple(map(lambda ix: {0: max, 1: min}[ix[0]](ix[1]), enumerate(zip(*toindices(patch)))))

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

def palette(
    element: Element
) -> IntegerSet:
    """ colors occurring in object or grid """
    if isinstance(element, tuple):
        return frozenset({v for r in element for v in r})
    return frozenset({v for v, _ in element})

def delta(
    patch: Patch
) -> Indices:
    """ indices in bounding box but not part of patch """
    if len(patch) == 0:
        return frozenset({})
    return backdrop(patch) - toindices(patch)

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_aba27056(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = fgpartition(I)
    x1 = merge(x0)
    x2 = delta(x1)
    x3 = fill(I, FOUR, x2)
    x4 = delta(x1)
    x5 = box(x1)
    x6 = intersection(x4, x5)
    x7 = uppermost(x6)
    x8 = uppermost(x1)
    x9 = equality(x7, x8)
    x10 = leftmost(x6)
    x11 = leftmost(x1)
    x12 = equality(x10, x11)
    x13 = lowermost(x6)
    x14 = lowermost(x1)
    x15 = equality(x13, x14)
    x16 = rightmost(x6)
    x17 = rightmost(x1)
    x18 = equality(x16, x17)
    x19 = urcorner(x6)
    x20 = ulcorner(x6)
    x21 = llcorner(x6)
    x22 = lrcorner(x6)
    x23 = branch(x15, x21, x22)
    x24 = branch(x12, x20, x23)
    x25 = branch(x9, x19, x24)
    x26 = branch(x15, x22, x19)
    x27 = branch(x12, x21, x26)
    x28 = branch(x9, x20, x27)
    x29 = branch(x15, DOWN_LEFT, UNITY)
    x30 = branch(x12, NEG_UNITY, x29)
    x31 = branch(x9, UP_RIGHT, x30)
    x32 = branch(x15, UNITY, UP_RIGHT)
    x33 = branch(x12, DOWN_LEFT, x32)
    x34 = branch(x9, NEG_UNITY, x33)
    x35 = branch(x15, DOWN, RIGHT)
    x36 = branch(x12, LEFT, x35)
    x37 = branch(x9, UP, x36)
    x38 = shoot(x25, x31)
    x39 = shoot(x28, x34)
    x40 = combine(x38, x39)
    x41 = rbind(shoot, x37)
    x42 = mapply(x41, x6)
    x43 = combine(x42, x40)
    x44 = fill(x3, FOUR, x43)
    return x44


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_aba27056(inp)
        assert pred == _to_grid(expected), f"{name} failed"

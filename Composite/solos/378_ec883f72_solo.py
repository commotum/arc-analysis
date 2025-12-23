# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "ec883f72"
SERIAL = "378"
URL    = "https://arcprize.org/play?task=ec883f72"

# --- Code Golf Concepts ---
CONCEPTS = [
    "pattern_expansion",
    "draw_line_from_point",
    "diagonals",
]

# --- Example Grids ---
E1_IN = np.array([
    [3, 3, 0, 9, 0, 0],
    [3, 3, 0, 9, 0, 0],
    [0, 0, 0, 9, 0, 0],
    [9, 9, 9, 9, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [3, 3, 0, 9, 0, 0],
    [3, 3, 0, 9, 0, 0],
    [0, 0, 0, 9, 0, 0],
    [9, 9, 9, 9, 0, 0],
    [0, 0, 0, 0, 3, 0],
    [0, 0, 0, 0, 0, 3],
], dtype=int)

E2_IN = np.array([
    [0, 0, 8, 0, 6, 0, 8, 0],
    [0, 0, 8, 0, 0, 0, 8, 0],
    [0, 0, 8, 8, 8, 8, 8, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [0, 0, 8, 0, 6, 0, 8, 0],
    [0, 0, 8, 0, 0, 0, 8, 0],
    [0, 0, 8, 8, 8, 8, 8, 0],
    [0, 6, 0, 0, 0, 0, 0, 6],
    [6, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E3_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 4, 4, 4, 4, 4, 4, 0, 0],
    [0, 4, 0, 0, 0, 0, 4, 0, 0],
    [0, 4, 0, 2, 2, 0, 4, 0, 0],
    [0, 4, 0, 2, 2, 0, 4, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 2],
    [2, 0, 0, 0, 0, 0, 0, 2, 0],
    [0, 4, 4, 4, 4, 4, 4, 0, 0],
    [0, 4, 0, 0, 0, 0, 4, 0, 0],
    [0, 4, 0, 2, 2, 0, 4, 0, 0],
    [0, 4, 0, 2, 2, 0, 4, 0, 0],
], dtype=int)

E4_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
    [4, 4, 4, 4, 0, 5, 0, 0, 0, 0, 0, 0],
    [4, 4, 4, 4, 0, 5, 0, 0, 0, 0, 0, 0],
    [4, 4, 4, 4, 0, 5, 0, 0, 0, 0, 0, 0],
    [4, 4, 4, 4, 0, 5, 0, 0, 0, 0, 0, 0],
    [4, 4, 4, 4, 0, 5, 0, 0, 0, 0, 0, 0],
], dtype=int)

E4_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0],
    [5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
    [4, 4, 4, 4, 0, 5, 0, 0, 0, 0, 0, 0],
    [4, 4, 4, 4, 0, 5, 0, 0, 0, 0, 0, 0],
    [4, 4, 4, 4, 0, 5, 0, 0, 0, 0, 0, 0],
    [4, 4, 4, 4, 0, 5, 0, 0, 0, 0, 0, 0],
    [4, 4, 4, 4, 0, 5, 0, 0, 0, 0, 0, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 0, 0, 3, 0, 4, 4, 0, 3, 0, 0],
    [0, 0, 0, 0, 3, 0, 4, 4, 0, 3, 0, 0],
    [0, 0, 0, 0, 3, 0, 4, 4, 0, 3, 0, 0],
    [0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0],
    [0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

T_OUT = np.array([
    [0, 0, 0, 0, 3, 0, 4, 4, 0, 3, 0, 0],
    [0, 0, 0, 0, 3, 0, 4, 4, 0, 3, 0, 0],
    [0, 0, 0, 0, 3, 0, 4, 4, 0, 3, 0, 0],
    [0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0],
    [0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0],
    [0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 4, 0],
    [0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 4],
    [0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def f(j,A,c,E,k):
 W=j[A][c]
 if W==0:return
 if not sum(j[A][c+i]==W for i in(1,-1))==sum(j[A+i][c]==W for i in(1,-1))==1:return
 l,J,p,a=2*(j[A+1][c]==W)-1,2*(j[A][c+1]==W)-1,c,A
 if j[A+l][c+J]==W:return
 while 1<=p<k-1 and 1<=a<E-1:a-=l;p-=J;j[a][p]=j[A+2*l][c+2*J]
def p(j):
 E,k=len(j),len(j[0])
 for A in range(1,E-1):
  for c in range(1,k-1):f(j,A,c,E,k)
 return j


# --- Code Golf Solution (Compressed) ---
def q(g):
    return exec("x='...'*len(g)+'.0';g[::-1]=zip(*eval(re.sub(f'0(?=({x}, .)*, [^0]{x*2}, (.))',r'\\2',str(g))));" * 4) or g


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import randint, sample, uniform

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

def interval(
    start: Integer,
    stop: Integer,
    step: Integer
) -> Tuple:
    """ range """
    return tuple(range(start, stop, step))

def ofcolor(
    grid: Grid,
    value: Integer
) -> Indices:
    """ indices of all grid cells with value """
    return frozenset((i, j) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value)

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

def llcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of lower left corner """
    return tuple(map(lambda ix: {0: max, 1: min}[ix[0]](ix[1]), enumerate(zip(*toindices(patch)))))

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

def outbox(
    patch: Patch
) -> Indices:
    """ outbox for patch """
    ai, aj = uppermost(patch) - 1, leftmost(patch) - 1
    bi, bj = lowermost(patch) + 1, rightmost(patch) + 1
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

def generate_ec883f72(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    ohi = unifint(diff_lb, diff_ub, (0, h - 6))
    owi = unifint(diff_lb, diff_ub, (0, w - 6))
    oh = h - 5 - ohi
    ow = w - 5 - owi
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    bgc, sqc, linc = sample(cols, 3)
    gi = canvas(bgc, (h, w))
    obj = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
    gi = fill(gi, sqc, obj)
    obob = outbox(outbox(obj))
    gi = fill(gi, linc, obob)
    ln1 = shoot(lrcorner(obob), (1, 1))
    ln2 = shoot(ulcorner(obob), (-1, -1))
    ln3 = shoot(llcorner(obob), (1, -1))
    ln4 = shoot(urcorner(obob), (-1, 1))
    lns = (ln1 | ln2 | ln3 | ln4) & ofcolor(gi, bgc)
    go = fill(gi, sqc, lns)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Numerical = Union[Integer, IntegerTuple]

IntegerSet = FrozenSet[Integer]

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

UNITY = (1, 1)

NEG_UNITY = (-1, -1)

UP_RIGHT = (-1, 1)

DOWN_LEFT = (1, -1)

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

def combine(
    a: Container,
    b: Container
) -> Container:
    """ union """
    return type(a)((*a, *b))

def argmax(
    container: Container,
    compfunc: Callable
) -> Any:
    """ largest item by custom order """
    return max(container, key=compfunc, default=None)

def first(
    container: Container
) -> Any:
    """ first item of container """
    return next(iter(container))

def remove(
    value: Any,
    container: Container
) -> Container:
    """ remove item from container """
    return type(container)(e for e in container if e != value)

def other(
    container: Container,
    value: Any
) -> Any:
    """ other value in the container """
    return first(remove(value, container))

def fork(
    outer: Callable,
    a: Callable,
    b: Callable
) -> Callable:
    """ creates a wrapper function """
    return lambda x: outer(a(x), b(x))

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

def partition(
    grid: Grid
) -> Objects:
    """ each cell with the same value part of the same object """
    return frozenset(
        frozenset(
            (v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value
        ) for value in palette(grid)
    )

def palette(
    element: Element
) -> IntegerSet:
    """ colors occurring in object or grid """
    if isinstance(element, tuple):
        return frozenset({v for r in element for v in r})
    return frozenset({v for v, _ in element})

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

def verify_ec883f72(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = fork(multiply, height, width)
    x1 = partition(I)
    x2 = argmax(x1, x0)
    x3 = remove(x2, x1)
    x4 = argmax(x3, x0)
    x5 = other(x3, x4)
    x6 = palette(I)
    x7 = lrcorner(x4)
    x8 = add(x7, UNITY)
    x9 = llcorner(x4)
    x10 = add(x9, DOWN_LEFT)
    x11 = urcorner(x4)
    x12 = add(x11, UP_RIGHT)
    x13 = ulcorner(x4)
    x14 = add(x13, NEG_UNITY)
    x15 = shoot(x8, UNITY)
    x16 = shoot(x10, DOWN_LEFT)
    x17 = shoot(x12, UP_RIGHT)
    x18 = shoot(x14, NEG_UNITY)
    x19 = combine(x15, x16)
    x20 = combine(x17, x18)
    x21 = combine(x19, x20)
    x22 = color(x5)
    x23 = fill(I, x22, x21)
    return x23


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("E4", E4_IN, E4_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_ec883f72(inp)
        assert pred == _to_grid(expected), f"{name} failed"

# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "d9fac9be"
SERIAL = "346"
URL    = "https://arcprize.org/play?task=d9fac9be"

# --- Code Golf Concepts ---
CONCEPTS = [
    "find_the_intruder",
    "summarize",
    "x_marks_the_spot",
]

# --- Example Grids ---
E1_IN = np.array([
    [2, 0, 0, 0, 0, 2, 0, 0, 2],
    [0, 4, 4, 4, 0, 0, 0, 0, 0],
    [0, 4, 2, 4, 0, 0, 2, 0, 0],
    [0, 4, 4, 4, 0, 0, 0, 2, 0],
    [2, 0, 0, 0, 0, 2, 0, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [2],
], dtype=int)

E2_IN = np.array([
    [8, 0, 8, 0, 0, 0, 0, 0, 8],
    [0, 0, 0, 0, 8, 0, 0, 0, 0],
    [0, 0, 8, 0, 0, 3, 3, 3, 0],
    [8, 0, 0, 3, 0, 3, 8, 3, 0],
    [0, 0, 0, 0, 0, 3, 3, 3, 0],
    [0, 0, 8, 0, 0, 0, 0, 0, 0],
    [3, 0, 0, 8, 0, 0, 0, 8, 0],
], dtype=int)

E2_OUT = np.array([
    [8],
], dtype=int)

E3_IN = np.array([
    [1, 2, 0, 0, 0, 2, 0, 0, 0],
    [0, 0, 2, 0, 0, 0, 0, 0, 0],
    [2, 0, 1, 2, 0, 2, 0, 1, 1],
    [0, 1, 0, 0, 2, 0, 0, 0, 2],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 2, 2, 0, 0, 0, 0, 0],
    [1, 2, 1, 2, 0, 0, 0, 2, 0],
    [0, 2, 2, 2, 0, 0, 0, 0, 2],
    [0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 0, 0, 0, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [1],
], dtype=int)

E4_IN = np.array([
    [0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 3, 8],
    [3, 0, 0, 0, 0, 0, 0, 8, 0, 3, 0, 0],
    [0, 3, 3, 8, 0, 0, 0, 0, 0, 0, 0, 8],
    [0, 0, 0, 3, 8, 0, 0, 0, 0, 0, 0, 0],
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0],
    [0, 0, 0, 3, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 3, 3, 3, 0, 0, 8, 0, 3, 0],
    [0, 0, 3, 3, 8, 3, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 0],
    [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E4_OUT = np.array([
    [8],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 4, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 4, 0, 0, 4, 0, 0, 0],
    [0, 4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 4, 4, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 4, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 4],
    [4, 0, 0, 0, 1, 4, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 4],
    [0, 0, 4, 4, 0, 0, 0, 1, 0, 0, 0, 0],
], dtype=int)

T_OUT = np.array([
    [4],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
from collections import*
def p(j):
 for A in range(0,len(j)-3+1,1):
  for c in range(0,len(j[0])-3+1,1):
   E=j[A:A+3];E=[R[c:c+3]for R in E];k=[i for s in E for i in s];W=Counter(k).most_common(1)
   if min(k)>0 and W[0][1]==8:return[[E[1][1]]]


# --- Code Golf Solution (Compressed) ---
def q(a):
    return [[min((b := sum(a[1:-1], a[3])), key=b.count)]]


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

def leftmost(
    patch: Patch
) -> Integer:
    """ column index of leftmost occupied cell """
    return min(j for i, j in toindices(patch))

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

def generate_d9fac9be(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    bgc, noisec, ringc = sample(cols, 3)
    gi = canvas(bgc, (h, w))
    nnoise1 = unifint(diff_lb, diff_ub, (1, (h * w) // 3 - 1))
    nnoise2 = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 3 - 9)))
    inds = asindices(gi)
    noise1 = sample(totuple(inds), nnoise1)
    noise2 = sample(difference(totuple(inds), noise1), nnoise2)
    gi = fill(gi, noisec, noise1)
    gi = fill(gi, ringc, noise2)
    rng = neighbors((1, 1))
    fp1 = recolor(noisec, rng)
    fp2 = recolor(ringc, rng)
    fp1occ = occurrences(gi, fp1)
    fp2occ = occurrences(gi, fp2)
    for occ1 in fp1occ:
        loc = choice(totuple(shift(rng, occ1)))
        gi = fill(gi, choice((bgc, ringc)), {loc})
    for occ2 in fp2occ:
        loc = choice(totuple(shift(rng, occ2)))
        gi = fill(gi, choice((bgc, noisec)), {loc})
    loci = randint(0, h - 3)
    locj = randint(0, w - 3)
    ringp = shift(rng, (loci, locj))
    gi = fill(gi, ringc, ringp)
    gi = fill(gi, noisec, {(loci + 1, locj + 1)})
    go = canvas(noisec, (1, 1))
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

IntegerSet = FrozenSet[Integer]

Element = Union[Object, Grid]

UNITY = (1, 1)

def combine(
    a: Container,
    b: Container
) -> Container:
    """ union """
    return type(a)((*a, *b))

def size(
    container: Container
) -> Integer:
    """ cardinality """
    return len(container)

def initset(
    value: Any
) -> FrozenSet:
    """ initialize container """
    return frozenset({value})

def positive(
    x: Integer
) -> Boolean:
    """ positive """
    return x > 0

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

def remove(
    value: Any,
    container: Container
) -> Container:
    """ remove item from container """
    return type(container)(e for e in container if e != value)

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

def palette(
    element: Element
) -> IntegerSet:
    """ colors occurring in object or grid """
    if isinstance(element, tuple):
        return frozenset({v for r in element for v in r})
    return frozenset({v for v, _ in element})

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_d9fac9be(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = mostcolor(I)
    x1 = palette(I)
    x2 = remove(x0, x1)
    x3 = totuple(x2)
    x4 = first(x3)
    x5 = last(x3)
    x6 = neighbors(UNITY)
    x7 = initset(UNITY)
    x8 = recolor(x4, x6)
    x9 = recolor(x5, x7)
    x10 = combine(x8, x9)
    x11 = occurrences(I, x10)
    x12 = size(x11)
    x13 = positive(x12)
    x14 = branch(x13, x5, x4)
    x15 = canvas(x14, UNITY)
    return x15


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("E4", E4_IN, E4_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_d9fac9be(inp)
        assert pred == _to_grid(expected), f"{name} failed"

# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "91714a58"
SERIAL = "222"
URL    = "https://arcprize.org/play?task=91714a58"

# --- Code Golf Concepts ---
CONCEPTS = [
    "find_the_intruder",
    "remove_noise",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 0, 0, 1, 1, 4, 0, 2, 0, 0, 0, 0, 2, 0, 5],
    [0, 0, 0, 3, 5, 0, 0, 0, 9, 9, 8, 0, 4, 0, 5, 8],
    [1, 0, 8, 2, 8, 0, 0, 6, 0, 8, 5, 0, 0, 0, 8, 0],
    [0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0],
    [0, 0, 1, 2, 2, 2, 0, 0, 1, 9, 5, 0, 0, 2, 0, 4],
    [0, 4, 0, 2, 2, 2, 0, 2, 0, 0, 7, 0, 0, 0, 0, 0],
    [3, 0, 6, 2, 2, 2, 0, 0, 0, 3, 5, 0, 7, 0, 0, 0],
    [7, 0, 4, 6, 0, 0, 4, 7, 7, 3, 0, 2, 0, 0, 7, 1],
    [0, 7, 0, 0, 0, 0, 0, 9, 7, 7, 0, 0, 0, 8, 5, 2],
    [1, 5, 6, 4, 9, 3, 0, 3, 0, 0, 0, 0, 0, 9, 4, 6],
    [0, 2, 4, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 6, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4],
    [0, 0, 6, 0, 0, 0, 0, 0, 6, 0, 0, 2, 0, 0, 0, 0],
    [0, 3, 0, 0, 7, 0, 2, 0, 7, 9, 0, 0, 0, 0, 0, 0],
    [0, 0, 5, 0, 7, 0, 0, 0, 0, 0, 0, 0, 6, 5, 3, 0],
    [1, 0, 0, 9, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 9, 0],
], dtype=int)

E1_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E2_IN = np.array([
    [0, 0, 7, 0, 0, 6, 0, 6, 0, 0, 0, 7, 3, 0, 0, 0],
    [0, 0, 3, 0, 0, 1, 0, 0, 8, 0, 0, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 3, 9, 0, 0, 0, 0, 0, 0, 0, 8, 0, 8],
    [2, 2, 0, 2, 9, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0],
    [0, 5, 2, 0, 0, 7, 0, 6, 0, 0, 0, 3, 0, 0, 1, 0],
    [4, 4, 0, 3, 9, 0, 0, 0, 0, 7, 0, 2, 0, 0, 0, 0],
    [8, 0, 0, 0, 0, 6, 0, 0, 0, 8, 0, 0, 3, 0, 0, 0],
    [0, 9, 0, 0, 0, 4, 8, 0, 0, 0, 7, 0, 0, 0, 0, 0],
    [0, 0, 9, 5, 0, 0, 0, 0, 4, 6, 0, 1, 4, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 0, 8, 0, 5, 9, 4],
    [0, 9, 3, 9, 0, 3, 0, 0, 5, 6, 7, 0, 5, 0, 0, 0],
    [0, 0, 6, 6, 6, 6, 6, 6, 6, 0, 0, 0, 0, 7, 0, 0],
    [0, 4, 6, 6, 6, 6, 6, 6, 6, 0, 0, 4, 4, 6, 0, 2],
    [0, 5, 0, 0, 0, 0, 4, 5, 3, 0, 8, 0, 0, 0, 6, 9],
    [0, 0, 9, 7, 5, 0, 0, 0, 0, 0, 0, 0, 1, 0, 7, 1],
    [0, 8, 0, 0, 0, 0, 0, 1, 0, 3, 0, 0, 3, 8, 7, 0],
], dtype=int)

E2_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 6, 6, 6, 6, 6, 6, 6, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 6, 6, 6, 6, 6, 6, 6, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E3_IN = np.array([
    [3, 0, 0, 0, 0, 0, 6, 2, 0, 0, 0, 5, 0, 0, 0, 3],
    [0, 7, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 5, 0],
    [0, 0, 0, 0, 0, 8, 8, 0, 7, 7, 7, 0, 0, 0, 0, 4],
    [0, 2, 0, 0, 0, 0, 0, 0, 7, 7, 7, 0, 2, 0, 5, 0],
    [0, 8, 0, 0, 9, 6, 1, 7, 7, 7, 7, 0, 0, 0, 0, 0],
    [5, 0, 0, 0, 0, 3, 6, 0, 6, 0, 0, 3, 3, 0, 0, 0],
    [0, 4, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0],
    [9, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 8, 0, 0, 0, 0],
    [0, 0, 3, 0, 0, 0, 0, 6, 0, 9, 0, 0, 0, 0, 0, 0],
    [9, 0, 0, 0, 1, 0, 0, 3, 0, 8, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 7, 0],
    [0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 5, 0, 0],
    [4, 0, 0, 1, 7, 0, 3, 0, 0, 7, 5, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 7, 2, 0, 0, 5, 0, 0, 1, 0, 4],
    [0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 2, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 7, 9, 0, 0, 0, 5, 0, 2, 0, 3, 0],
], dtype=int)

E3_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 1, 7, 3, 0, 0, 0, 0, 0, 1, 2, 0, 4, 7, 0],
    [0, 0, 0, 3, 0, 0, 6, 8, 0, 0, 0, 0, 0, 0, 0, 0],
    [6, 0, 0, 8, 0, 1, 0, 0, 1, 0, 0, 0, 7, 0, 4, 8],
    [0, 3, 8, 0, 0, 0, 3, 0, 8, 0, 0, 0, 0, 0, 0, 0],
    [5, 0, 0, 0, 1, 0, 0, 8, 0, 0, 3, 8, 0, 0, 5, 0],
    [0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 3, 7, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 5, 0, 7],
    [0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 2, 7, 0, 7, 0, 0],
    [9, 4, 0, 2, 1, 0, 0, 0, 0, 0, 7, 0, 0, 0, 9, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 5],
    [0, 8, 9, 4, 0, 5, 5, 5, 5, 5, 5, 3, 0, 0, 0, 0],
    [0, 0, 3, 0, 6, 5, 5, 5, 5, 5, 5, 0, 1, 4, 0, 0],
    [9, 5, 2, 0, 0, 5, 1, 3, 0, 0, 6, 2, 0, 0, 1, 5],
    [0, 7, 0, 0, 0, 0, 1, 6, 0, 7, 0, 3, 0, 6, 0, 0],
    [0, 0, 9, 0, 0, 3, 7, 7, 0, 6, 0, 0, 8, 0, 0, 0],
    [5, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 9],
], dtype=int)

T_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
E=enumerate
def p(g):
 g=[[v if(i and g[i-1][j]==v)+(i+1<len(g)and g[i+1][j]==v)+(j and r[j-1]==v)+(j+1<len(g)and r[j+1]==v)>1else 0 for j,v in E(r)]for i,r in E(g)]
 f=sum(g,[])
 C=sorted([[f.count(c),c] for c in set(f) if c>0])
 g=[[0 if c!=C[-1][1] else c for c in r] for r in g]
 return g


# --- Code Golf Solution (Compressed) ---
def q(g):
    return [(g := [[c * (str(r * 7 + g).count(2 * f'{c}, ') > 9) for c in r] for *r, in zip(*g)]) for _ in g][5]


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

def generate_91714a58(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    bgc, targc = sample(cols, 2)
    remcols = remove(bgc, cols)
    nnoise = unifint(diff_lb, diff_ub, (1, (h * w) // 2))
    gi = canvas(bgc, (h, w))
    inds = totuple(asindices(gi))
    noise = sample(inds, nnoise)
    ih = randint(2, h // 2)
    iw = randint(2, w // 2)
    loci = randint(0, h - ih)
    locj = randint(0, w - iw)
    loc = (loci, locj)
    bd = backdrop(frozenset({(loci, locj), (loci + ih - 1, locj + iw - 1)}))
    go = fill(gi, targc, bd)
    for ij in noise:
        col = choice(remcols)
        gi = fill(gi, col, {ij})
    gi = fill(gi, targc, bd)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

ContainerContainer = Container[Container]

F = False

T = True

ORIGIN = (0, 0)

RIGHT = (0, 1)

ZERO_BY_TWO = (0, 2)

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

def merge(
    containers: ContainerContainer
) -> Container:
    """ merging """
    return type(containers)(e for c in containers for e in c)

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

def color(
    obj: Object
) -> Integer:
    """ color of object """
    return next(iter(obj))[0]

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

def verify_91714a58(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = shape(I)
    x1 = asindices(I)
    x2 = objects(I, T, F, T)
    x3 = argmax(x2, size)
    x4 = mostcolor(x3)
    x5 = mostcolor(I)
    x6 = canvas(x5, x0)
    x7 = paint(x6, x3)
    x8 = mostcolor(I)
    x9 = color(x3)
    x10 = astuple(x8, ORIGIN)
    x11 = astuple(x9, RIGHT)
    x12 = astuple(x8, ZERO_BY_TWO)
    x13 = initset(x12)
    x14 = insert(x11, x13)
    x15 = insert(x10, x14)
    x16 = dmirror(x15)
    x17 = toindices(x15)
    x18 = lbind(shift, x17)
    x19 = occurrences(x7, x15)
    x20 = mapply(x18, x19)
    x21 = toindices(x16)
    x22 = lbind(shift, x21)
    x23 = occurrences(x7, x16)
    x24 = mapply(x22, x23)
    x25 = combine(x20, x24)
    x26 = fill(x7, x8, x25)
    return x26


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_91714a58(inp)
        assert pred == _to_grid(expected), f"{name} failed"

# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "4be741c5"
SERIAL = "115"
URL    = "https://arcprize.org/play?task=4be741c5"

# --- Code Golf Concepts ---
CONCEPTS = [
    "summarize",
]

# --- Example Grids ---
E1_IN = np.array([
    [4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 8, 8],
    [4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 8, 8, 8, 8],
    [4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 8, 8, 8, 8],
    [4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 8, 8, 8, 8],
    [4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 8, 8, 8],
    [4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 8, 8, 8, 8],
    [4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 8, 8, 8, 8],
    [4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 8, 8, 8],
    [4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 8, 8],
    [4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 8, 8],
    [4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 8, 8, 8],
    [4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 8, 8, 8, 8],
    [4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 8, 8, 8, 8],
    [4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 8, 8, 8],
], dtype=int)

E1_OUT = np.array([
    [4, 2, 8],
], dtype=int)

E2_IN = np.array([
    [2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2],
    [2, 8, 8, 8, 2, 2, 8],
    [8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 5, 5, 8, 8],
    [5, 8, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5, 5, 5],
], dtype=int)

E2_OUT = np.array([
    [2],
    [8],
    [5],
], dtype=int)

E3_IN = np.array([
    [6, 6, 6, 6, 6, 6, 6, 6, 6],
    [6, 6, 4, 4, 6, 6, 6, 6, 6],
    [6, 4, 4, 4, 6, 4, 6, 4, 4],
    [4, 4, 4, 4, 4, 4, 4, 4, 4],
    [4, 4, 4, 4, 4, 4, 4, 4, 4],
    [4, 4, 4, 4, 4, 4, 4, 4, 4],
    [4, 2, 2, 4, 4, 4, 2, 2, 4],
    [2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 3, 2, 2, 2, 2, 2, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3],
], dtype=int)

E3_OUT = np.array([
    [6],
    [4],
    [2],
    [3],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 8, 8],
    [3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 8, 8, 8],
    [3, 3, 3, 3, 3, 2, 2, 1, 1, 1, 8, 8, 8, 8],
    [3, 3, 3, 3, 3, 2, 2, 1, 1, 1, 1, 8, 8, 8],
    [3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 8, 8],
    [3, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 8],
    [3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 1, 8, 8],
    [3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 8, 8, 8],
    [3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 8, 8],
    [3, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 8, 8],
    [3, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1, 8, 8, 8],
    [3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 8, 8],
    [3, 3, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 8, 8],
    [3, 3, 3, 3, 3, 3, 2, 2, 2, 1, 1, 1, 8, 8],
], dtype=int)

T_OUT = np.array([
    [3, 2, 1, 8],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j):
 def u(A):
  c=[]
  for E in A:
   if E not in c:c.append(E)
  return c
 k=[u(c)for c in j]
 if all(k[0]==c for c in k):return[k[0]]
 return[[E]for E in u([E for c in j for E in c])]


# --- Code Golf Solution (Compressed) ---
def q(g, F={}.fromkeys):
    return [*F(zip(*zip(*map(F, g))))]


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

def repeat(
    item: Any,
    num: Integer
) -> Tuple:
    """ repetition of item within vector """
    return tuple(item for i in range(num))

def merge(
    containers: ContainerContainer
) -> Container:
    """ merging """
    return type(containers)(e for c in containers for e in c)

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

def generate_4be741c5(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    numcolors = unifint(diff_lb, diff_ub, (2, w // 3))
    ccols = sample(cols, numcolors)
    go = (tuple(ccols),)
    gi = merge(tuple(repeat(repeat(c, h), 3) for c in ccols))
    while len(gi) < w:
        idx = randint(0, len(gi) - 1)
        gi = gi[:idx] + gi[idx:idx+1] + gi[idx:]
    gi = dmirror(gi)
    ndisturbances = unifint(diff_lb, diff_ub, (0, 3 * h * numcolors))
    for k in range(ndisturbances):
        options = []
        for a in range(h):
            for b in range(w - 3):
                if gi[a][b] == gi[a][b+1] and gi[a][b+2] == gi[a][b+3]:
                    options.append((a, b, gi[a][b], gi[a][b+2]))
        if len(options) == 0:
            break
        a, b, c1, c2 = choice(options)
        if choice((True, False)):
            gi = fill(gi, c2, {(a, b+1)})
        else:
            gi = fill(gi, c1, {(a, b+2)})
    if choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

ONE = 1

ORIGIN = (0, 0)

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

def apply(
    function: Callable,
    container: Container
) -> Container:
    """ apply function to each item in container """
    return type(container)(function(e) for e in container)

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

def crop(
    grid: Grid,
    start: IntegerTuple,
    dims: IntegerTuple
) -> Grid:
    """ subgrid specified by start and dimension """
    return tuple(r[start[1]:start[1]+dims[1]] for r in grid[start[0]:start[0]+dims[0]])

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

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_4be741c5(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = first(I)
    x1 = dedupe(x0)
    x2 = size(x1)
    x3 = equality(x2, ONE)
    x4 = branch(x3, dmirror, identity)
    x5 = branch(x3, height, width)
    x6 = x5(I)
    x7 = astuple(ONE, x6)
    x8 = x4(I)
    x9 = crop(x8, ORIGIN, x7)
    x10 = apply(dedupe, x9)
    x11 = x4(x10)
    return x11


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_4be741c5(inp)
        assert pred == _to_grid(expected), f"{name} failed"

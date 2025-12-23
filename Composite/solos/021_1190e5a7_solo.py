# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "1190e5a7"
SERIAL = "021"
URL    = "https://arcprize.org/play?task=1190e5a7"

# --- Code Golf Concepts ---
CONCEPTS = [
    "detect_grid",
    "count_hor_lines",
    "count_ver_lines",
    "detect_background_color",
    "color_guessing",
    "create_image_from_info",
]

# --- Example Grids ---
E1_IN = np.array([
    [3, 7, 3, 3, 3, 3, 3, 3, 3, 3, 7, 3, 3, 7, 3],
    [3, 7, 3, 3, 3, 3, 3, 3, 3, 3, 7, 3, 3, 7, 3],
    [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
    [3, 7, 3, 3, 3, 3, 3, 3, 3, 3, 7, 3, 3, 7, 3],
    [3, 7, 3, 3, 3, 3, 3, 3, 3, 3, 7, 3, 3, 7, 3],
    [3, 7, 3, 3, 3, 3, 3, 3, 3, 3, 7, 3, 3, 7, 3],
    [3, 7, 3, 3, 3, 3, 3, 3, 3, 3, 7, 3, 3, 7, 3],
    [3, 7, 3, 3, 3, 3, 3, 3, 3, 3, 7, 3, 3, 7, 3],
    [3, 7, 3, 3, 3, 3, 3, 3, 3, 3, 7, 3, 3, 7, 3],
    [3, 7, 3, 3, 3, 3, 3, 3, 3, 3, 7, 3, 3, 7, 3],
    [3, 7, 3, 3, 3, 3, 3, 3, 3, 3, 7, 3, 3, 7, 3],
    [3, 7, 3, 3, 3, 3, 3, 3, 3, 3, 7, 3, 3, 7, 3],
    [3, 7, 3, 3, 3, 3, 3, 3, 3, 3, 7, 3, 3, 7, 3],
    [3, 7, 3, 3, 3, 3, 3, 3, 3, 3, 7, 3, 3, 7, 3],
    [3, 7, 3, 3, 3, 3, 3, 3, 3, 3, 7, 3, 3, 7, 3],
], dtype=int)

E1_OUT = np.array([
    [3, 3, 3, 3],
    [3, 3, 3, 3],
], dtype=int)

E2_IN = np.array([
    [1, 1, 1, 1, 8, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 8, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 8, 1, 1, 1, 1, 1, 1],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [1, 1, 1, 1, 8, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 8, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 8, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 8, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 8, 1, 1, 1, 1, 1, 1],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [1, 1, 1, 1, 8, 1, 1, 1, 1, 1, 1],
], dtype=int)

E2_OUT = np.array([
    [1, 1],
    [1, 1],
    [1, 1],
], dtype=int)

E3_IN = np.array([
    [3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 1, 3, 1, 3],
    [3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 1, 3, 1, 3],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 1, 3, 1, 3],
    [3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 1, 3, 1, 3],
    [3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 1, 3, 1, 3],
    [3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 1, 3, 1, 3],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 1, 3, 1, 3],
    [3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 1, 3, 1, 3],
    [3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 1, 3, 1, 3],
    [3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 1, 3, 1, 3],
    [3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 1, 3, 1, 3],
    [3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 1, 3, 1, 3],
    [3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 1, 3, 1, 3],
    [3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 1, 3, 1, 3],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 1, 3, 1, 3],
    [3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 1, 3, 1, 3],
    [3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 1, 3, 1, 3],
    [3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 1, 3, 1, 3],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 1, 3, 1, 3],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 1, 3, 1, 3],
    [3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 1, 3, 1, 3],
    [3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 1, 3, 1, 3],
], dtype=int)

E3_OUT = np.array([
    [3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 5, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 5, 1],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 5, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 5, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 5, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 5, 1],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 5, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 5, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 5, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 5, 1],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 5, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 5, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 5, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 5, 1],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 5, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 5, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 5, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 5, 1],
], dtype=int)

T_OUT = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(g,u=range):n=len(g);m=len(g[0]);r=[i for i in u(n)if len(set(g[i]))==1];c=[j for j in u(m)if len(set(g[i][j]for i in u(n)))==1];b=next(x for i in u(n)for j,x in enumerate(g[i])if i not in r and j not in c);return[[b]*(len(c)+1)for _ in u(len(r)+1)]


# --- Code Golf Solution (Compressed) ---
def q(g):
    return g * -1 * -1 or [p(g[0])] * -~min(map(g.count, g))


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, uniform

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

ContainerContainer = Container[Container]

def combine(
    a: Container,
    b: Container
) -> Container:
    """ union """
    return type(a)((*a, *b))

def difference(
    a: Container,
    b: Container
) -> Container:
    """ difference """
    return type(a)(e for e in a if e not in b)

def merge(
    containers: ContainerContainer
) -> Container:
    """ merging """
    return type(containers)(e for c in containers for e in c)

def toivec(
    i: Integer
) -> IntegerTuple:
    """ vector pointing vertically """
    return (i, 0)

def tojvec(
    j: Integer
) -> IntegerTuple:
    """ vector pointing horizontally """
    return (0, j)

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

def chain(
    h: Callable,
    g: Callable,
    f: Callable
) -> Callable:
    """ function composition with three functions """
    return lambda x: h(g(f(x)))

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

def vfrontier(
    location: IntegerTuple
) -> Indices:
    """ vertical frontier """
    return frozenset((i, location[1]) for i in range(30))

def hfrontier(
    location: IntegerTuple
) -> Indices:
    """ horizontal frontier """
    return frozenset((location[0], j) for j in range(30))

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

def generate_1190e5a7(diff_lb: float, diff_ub: float) -> dict:
    dim_bounds = (3, 30)
    colopts = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, dim_bounds)
    w = unifint(diff_lb, diff_ub, dim_bounds)
    bgc = choice(colopts)
    c = canvas(bgc, (h, w))
    nhf_bounds = (1, h // 3)
    nvf_bounds = (1, w // 3)
    nhf = unifint(diff_lb, diff_ub, nhf_bounds)
    nvf = unifint(diff_lb, diff_ub, nvf_bounds)
    hf_options = interval(1, h - 1, 1)
    vf_options = interval(1, w - 1, 1)
    hf_selection = []
    for k in range(nhf):
        hf = choice(hf_options)
        hf_selection.append(hf)
        hf_options = difference(hf_options, (hf - 1, hf, hf + 1))
    vf_selection = []
    for k in range(nvf):
        vf = choice(vf_options)
        vf_selection.append(vf)
        vf_options = difference(vf_options, (vf - 1, vf, vf + 1))
    remcols = remove(bgc, colopts)
    rcf = lambda x: recolor(choice(remcols), x)
    hfs = mapply(chain(rcf, hfrontier, toivec), tuple(hf_selection))
    vfs = mapply(chain(rcf, vfrontier, tojvec), tuple(vf_selection))
    gi = paint(c, combine(hfs, vfs))
    go = canvas(bgc, (nhf + 1, nvf + 1))
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Numerical = Union[Integer, IntegerTuple]

IntegerSet = FrozenSet[Integer]

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

def equality(
    a: Any,
    b: Any
) -> Boolean:
    """ equality """
    return a == b

def size(
    container: Container
) -> Integer:
    """ cardinality """
    return len(container)

def argmin(
    container: Container,
    compfunc: Callable
) -> Any:
    """ smallest item by custom order """
    return min(container, key=compfunc, default=None)

def increment(
    x: Numerical
) -> Numerical:
    """ incrementing """
    return x + 1 if isinstance(x, int) else (x[0] + 1, x[1] + 1)

def sfilter(
    container: Container,
    condition: Callable
) -> Container:
    """ keep elements in container that satisfy condition """
    return type(container)(e for e in container if condition(e))

def astuple(
    a: Integer,
    b: Integer
) -> IntegerTuple:
    """ constructs a tuple """
    return (a, b)

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

def vline(
    patch: Patch
) -> Boolean:
    """ whether the piece forms a vertical line """
    return height(patch) == len(patch) and width(patch) == 1

def palette(
    element: Element
) -> IntegerSet:
    """ colors occurring in object or grid """
    if isinstance(element, tuple):
        return frozenset({v for r in element for v in r})
    return frozenset({v for v, _ in element})

def toobject(
    patch: Patch,
    grid: Grid
) -> Object:
    """ object from patch and grid """
    h, w = len(grid), len(grid[0])
    return frozenset((grid[i][j], (i, j)) for i, j in toindices(patch) if 0 <= i < h and 0 <= j < w)

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

def corners(
    patch: Patch
) -> Indices:
    """ indices of corners """
    return frozenset({ulcorner(patch), urcorner(patch), llcorner(patch), lrcorner(patch)})

def frontiers(
    grid: Grid
) -> Objects:
    """ set of frontiers """
    h, w = len(grid), len(grid[0])
    row_indices = tuple(i for i, r in enumerate(grid) if len(set(r)) == 1)
    column_indices = tuple(j for j, c in enumerate(dmirror(grid)) if len(set(c)) == 1)
    hfrontiers = frozenset({frozenset({(grid[i][j], (i, j)) for j in range(w)}) for i in row_indices})
    vfrontiers = frozenset({frozenset({(grid[i][j], (i, j)) for i in range(h)}) for j in column_indices})
    return hfrontiers | vfrontiers

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_1190e5a7(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = asindices(I)
    x1 = corners(x0)
    x2 = toobject(x1, I)
    x3 = mostcolor(x2)
    x4 = palette(I)
    x5 = rbind(equality, x3)
    x6 = argmin(x4, x5)
    x7 = asindices(I)
    x8 = ofcolor(I, x3)
    x9 = difference(x7, x8)
    x10 = fill(I, x6, x9)
    x11 = frontiers(x10)
    x12 = sfilter(x11, vline)
    x13 = difference(x11, x12)
    x14 = astuple(x13, x12)
    x15 = apply(size, x14)
    x16 = increment(x15)
    x17 = canvas(x3, x16)
    return x17


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_1190e5a7(inp)
        assert pred == _to_grid(expected), f"{name} failed"

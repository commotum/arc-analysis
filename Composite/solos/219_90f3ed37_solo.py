# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "90f3ed37"
SERIAL = "219"
URL    = "https://arcprize.org/play?task=90f3ed37"

# --- Code Golf Concepts ---
CONCEPTS = [
    "pattern_repetition",
    "recoloring",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 8, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 8, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 8, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 8, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 8, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 8, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E2_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 8, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 8, 8, 8, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 8, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 8, 8, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 8, 8, 8, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 8, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E3_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 0, 8, 0, 8, 0, 8, 0, 8, 0],
    [0, 8, 0, 8, 0, 8, 0, 8, 0, 8],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 0, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 8, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 0, 8, 0, 8, 0, 0, 0, 0, 0],
    [0, 8, 0, 8, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 0, 8, 0, 8, 0, 8, 0, 8, 0],
    [0, 8, 0, 8, 0, 8, 0, 8, 0, 8],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 0, 8, 0, 1, 0, 1, 0, 1, 0],
    [0, 8, 0, 1, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 0, 8, 0, 8, 0, 1, 0, 1, 0],
    [0, 8, 0, 8, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 8, 0, 0, 0, 0, 0],
    [8, 8, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 8, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

T_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 8, 1, 1, 1, 1, 1],
    [8, 8, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 8, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(*args, **kwargs):
    raise NotImplementedError("Barnacles solution not available for 219")


# --- Code Golf Solution (Compressed) ---
def q(_):
    m, *f = ([],)
    for e, r in enumerate(_):
        if max(r) < 1 and f:
            m += ({*f},)
            f = []
        for a, r in enumerate(r):
            f += [(a, e)] * r
    for d in m:
        for a, e in max([{(a + _, e - min(m[0])[1] + min(d)[1]) for a, e in m[0]} for _ in (2, 1, 0, -1)], key=d.__and__):
            if 0 < a < 10:
                _[e][a] += _[e][a] < 1
    return _


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, randint, sample, uniform

Integer = int

IntegerTuple = Tuple[Integer, Integer]

IntegerSet = FrozenSet[Integer]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

Element = Union[Object, Grid]

def difference(
    a: Container,
    b: Container
) -> Container:
    """ difference """
    return type(a)(e for e in a if e not in b)

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

def generate_90f3ed37(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(1, interval(0, 10, 1))
    while True:
        h = unifint(diff_lb, diff_ub, (8, 30))
        w = unifint(diff_lb, diff_ub, (8, 30))
        pathh = unifint(diff_lb, diff_ub, (1, max(1, h//4)))
        pathh = unifint(diff_lb, diff_ub, (pathh, max(1, h//4)))
        Lpatper = unifint(diff_lb, diff_ub, (1, w//7))
        Rpatper = unifint(diff_lb, diff_ub, (1, w//7))
        hh = randint(1, pathh)
        Linds = asindices(canvas(-1, (hh, Lpatper)))
        Rinds = asindices(canvas(-1, (hh, Rpatper)))
        lpatsd = unifint(diff_lb, diff_ub, (0, (hh * Lpatper) // 2))
        rpatsd = unifint(diff_lb, diff_ub, (0, (hh * Rpatper) // 2))
        lpats = choice((lpatsd, hh * Lpatper - lpatsd))
        rpats = choice((rpatsd, hh * Rpatper - rpatsd))
        lpats = min(max(Lpatper, lpats), hh * Lpatper)
        rpats = min(max(Rpatper, rpats), hh * Rpatper)
        lpat = set(sample(totuple(Linds), lpats))
        rpat = set(sample(totuple(Rinds), rpats))
        midpatw = randint(0, w-2*Lpatper-2*Rpatper)
        if midpatw == 0 or Lpatper == hh == 1:
            midpat = set()
            midpatw = 0
        else:
            midpat = set(sample(totuple(asindices(canvas(-1, (hh, midpatw)))), randint(midpatw, (hh * midpatw))))
        if shift(midpat, (0, 2*Lpatper-midpatw)).issubset(lpat):
            midpat = set()
            midpatw = 0
        loci = randint(0, h - pathh)
        lplac = shift(lpat, (loci, 0)) | shift(lpat, (loci, Lpatper))
        mplac = shift(midpat, (loci, 2*Lpatper))
        rplac = shift(rpat, (loci, 2*Lpatper+midpatw)) | shift(rpat, (loci, 2*Lpatper+midpatw+Rpatper))
        sp = 2*Lpatper+midpatw+Rpatper
        for k in range(w//Lpatper+1):
            lplac |= shift(lpat, (loci, -k*Lpatper))
        for k in range(w//Rpatper+1):
            rplac |= shift(rpat, (loci, sp+k*Rpatper))
        pat = lplac | mplac | rplac
        patn = shift(pat, (-loci, 0))
        bgc, fgc = sample(cols, 2)
        gi = canvas(bgc, (h, w))
        gi = fill(gi, fgc, pat)
        options = interval(0, h - pathh + 1, 1)
        options = difference(options, interval(loci-pathh-1, loci+2*pathh, 1))
        nplacements = unifint(diff_lb, diff_ub, (1, max(1, len(options) // pathh)))
        go = tuple(e for e in gi)
        for k in range(nplacements):
            if len(options) == 0:
                break
            locii = choice(options)
            options = difference(options, interval(locii-pathh-1, locii+2*pathh, 1))
            hoffs = randint(0, max(Rpatper, w-sp-2))
            cutoffopts = interval(2*Lpatper+midpatw, 2*Lpatper+midpatw+hoffs+1, 1)
            cutoffopts = cutoffopts[::-1]
            idx = unifint(diff_lb, diff_ub, (0, len(cutoffopts) - 1))
            cutoff = cutoffopts[idx]
            patnc = sfilter(patn, lambda ij: ij[1] <= cutoff)
            go = fill(go, 1, shift(patn, (locii, hoffs)))
            gi = fill(gi, fgc, shift(patnc, (locii, hoffs)))
            go = fill(go, fgc, shift(patnc, (locii, hoffs)))
        if 1 in palette(go):
            break
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Numerical = Union[Integer, IntegerTuple]

Piece = Union[Grid, Patch]

ContainerContainer = Container[Container]

ZERO = 0

ONE = 1

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

def invert(
    n: Numerical
) -> Numerical:
    """ inversion with respect to addition """
    return -n if isinstance(n, int) else (-n[0], -n[1])

def double(
    n: Numerical
) -> Numerical:
    """ scaling by two """
    return n * 2 if isinstance(n, int) else (n[0] * 2, n[1] * 2)

def contained(
    value: Any,
    container: Container
) -> Boolean:
    """ element of """
    return value in container

def intersection(
    a: FrozenSet,
    b: FrozenSet
) -> FrozenSet:
    """ returns the intersection of two containers """
    return a & b

def greater(
    a: Integer,
    b: Integer
) -> Boolean:
    """ greater """
    return a > b

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

def maximum(
    container: IntegerSet
) -> Integer:
    """ maximum """
    return max(container, default=0)

def minimum(
    container: IntegerSet
) -> Integer:
    """ minimum """
    return min(container, default=0)

def valmax(
    container: Container,
    compfunc: Callable
) -> Integer:
    """ maximum by custom function """
    return compfunc(max(container, key=compfunc, default=0))

def argmax(
    container: Container,
    compfunc: Callable
) -> Any:
    """ largest item by custom order """
    return max(container, key=compfunc, default=None)

def increment(
    x: Numerical
) -> Numerical:
    """ incrementing """
    return x + 1 if isinstance(x, int) else (x[0] + 1, x[1] + 1)

def positive(
    x: Integer
) -> Boolean:
    """ positive """
    return x > 0

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

def first(
    container: Container
) -> Any:
    """ first item of container """
    return next(iter(container))

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

def fork(
    outer: Callable,
    a: Callable,
    b: Callable
) -> Callable:
    """ creates a wrapper function """
    return lambda x: outer(a(x), b(x))

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

def leastcolor(
    element: Element
) -> Integer:
    """ least common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return min(set(values), key=values.count)

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

def center(
    patch: Patch
) -> IntegerTuple:
    """ center of the patch """
    return (uppermost(patch) + height(patch) // 2, leftmost(patch) + width(patch) // 2)

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_90f3ed37(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = leastcolor(I)
    x1 = ofcolor(I, x0)
    x2 = apply(first, x1)
    x3 = asindices(I)
    x4 = apply(first, x3)
    x5 = difference(x4, x2)
    x6 = ofcolor(I, x0)
    x7 = rbind(interval, ONE)
    x8 = lbind(rbind, contained)
    x9 = lbind(sfilter, x5)
    x10 = rbind(matcher, ZERO)
    x11 = chain(size, x9, x8)
    x12 = lbind(sfilter, x6)
    x13 = lbind(compose, x11)
    x14 = chain(x12, x10, x13)
    x15 = lbind(fork, x7)
    x16 = compose(increment, minimum)
    x17 = lbind(lbind, astuple)
    x18 = lbind(chain, x16)
    x19 = rbind(x18, first)
    x20 = chain(x19, x17, first)
    x21 = lbind(chain, maximum)
    x22 = rbind(x21, first)
    x23 = chain(x22, x17, first)
    x24 = fork(x15, x20, x23)
    x25 = compose(x14, x24)
    x26 = apply(toivec, x2)
    x27 = apply(x25, x26)
    x28 = argmax(x27, width)
    x29 = remove(x28, x27)
    x30 = ulcorner(x28)
    x31 = invert(x30)
    x32 = shift(x28, x31)
    x33 = asindices(I)
    x34 = center(x33)
    x35 = invert(x34)
    x36 = shift(x33, x35)
    x37 = width(I)
    x38 = double(x37)
    x39 = tojvec(x38)
    x40 = rbind(apply, x36)
    x41 = lbind(rbind, add)
    x42 = chain(x40, x41, center)
    x43 = compose(positive, size)
    x44 = lbind(compose, size)
    x45 = lbind(shift, x32)
    x46 = rbind(compose, x45)
    x47 = lbind(rbind, intersection)
    x48 = compose(x46, x47)
    x49 = lbind(compose, x43)
    x50 = compose(x49, x48)
    x51 = fork(sfilter, x42, x50)
    x52 = compose(x44, x48)
    x53 = fork(valmax, x51, x52)
    x54 = compose(x44, x48)
    x55 = fork(matcher, x54, x53)
    x56 = fork(sfilter, x51, x55)
    x57 = lbind(shift, x32)
    x58 = lbind(insert, x39)
    x59 = lbind(rbind, greater)
    x60 = compose(x59, rightmost)
    x61 = compose(leftmost, x58)
    x62 = rbind(compose, x57)
    x63 = lbind(rbind, difference)
    x64 = compose(x62, x63)
    x65 = lbind(compose, x61)
    x66 = compose(x65, x64)
    x67 = fork(compose, x60, x66)
    x68 = fork(argmax, x56, x67)
    x69 = lbind(shift, x32)
    x70 = compose(x69, x68)
    x71 = fork(difference, x70, identity)
    x72 = mapply(x71, x29)
    x73 = fill(I, ONE, x72)
    return x73


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_90f3ed37(inp)
        assert pred == _to_grid(expected), f"{name} failed"

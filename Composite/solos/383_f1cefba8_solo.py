# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "f1cefba8"
SERIAL = "383"
URL    = "https://arcprize.org/play?task=f1cefba8"

# --- Code Golf Concepts ---
CONCEPTS = [
    "draw_line_from_point",
    "pattern_modification",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0],
    [0, 8, 8, 8, 8, 8, 8, 2, 8, 8, 8, 8, 0],
    [0, 8, 8, 2, 2, 2, 2, 2, 2, 2, 8, 8, 0],
    [0, 8, 8, 2, 2, 2, 2, 2, 2, 2, 8, 8, 0],
    [0, 8, 8, 2, 2, 2, 2, 2, 2, 2, 2, 8, 0],
    [0, 8, 8, 2, 2, 2, 2, 2, 2, 2, 8, 8, 0],
    [0, 8, 8, 2, 2, 2, 2, 2, 2, 2, 8, 8, 0],
    [0, 8, 8, 2, 2, 2, 2, 2, 2, 2, 8, 8, 0],
    [0, 8, 8, 2, 2, 2, 2, 2, 2, 2, 8, 8, 0],
    [0, 8, 8, 2, 2, 2, 2, 2, 2, 2, 8, 8, 0],
    [0, 8, 8, 2, 2, 2, 2, 2, 2, 2, 8, 8, 0],
    [0, 8, 8, 8, 2, 8, 8, 8, 8, 8, 8, 8, 0],
    [0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0],
    [0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0],
    [0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0],
    [0, 8, 8, 2, 8, 2, 2, 8, 2, 2, 8, 8, 0],
    [0, 8, 8, 2, 8, 2, 2, 8, 2, 2, 8, 8, 0],
    [2, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 2],
    [0, 8, 8, 2, 8, 2, 2, 8, 2, 2, 8, 8, 0],
    [0, 8, 8, 2, 8, 2, 2, 8, 2, 2, 8, 8, 0],
    [0, 8, 8, 2, 8, 2, 2, 8, 2, 2, 8, 8, 0],
    [0, 8, 8, 2, 8, 2, 2, 8, 2, 2, 8, 8, 0],
    [0, 8, 8, 2, 8, 2, 2, 8, 2, 2, 8, 8, 0],
    [0, 8, 8, 2, 8, 2, 2, 8, 2, 2, 8, 8, 0],
    [0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0],
    [0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0],
    [0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0],
], dtype=int)

E2_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 0, 0, 0],
    [0, 0, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 4, 4, 1, 4, 1, 4, 4, 4, 1, 1, 0, 0, 0],
    [4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4],
    [0, 0, 1, 1, 4, 4, 1, 4, 1, 4, 4, 4, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 4, 4, 1, 4, 1, 4, 4, 4, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E3_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
    [0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 0, 0],
    [0, 0, 0, 0, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 0, 0],
    [0, 0, 0, 0, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 0, 0],
    [0, 0, 0, 0, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 0, 0],
    [0, 0, 0, 0, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 0, 0],
    [0, 0, 0, 0, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 0, 0],
    [0, 0, 0, 0, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 0, 0],
    [0, 0, 0, 0, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 0, 0],
    [0, 0, 0, 0, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 0, 0],
    [0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
    [0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
    [0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
    [0, 0, 0, 0, 2, 2, 3, 3, 3, 3, 3, 2, 3, 3, 2, 2, 0, 0],
    [0, 0, 0, 0, 2, 2, 3, 3, 3, 3, 3, 2, 3, 3, 2, 2, 0, 0],
    [0, 0, 0, 0, 2, 2, 3, 3, 3, 3, 3, 2, 3, 3, 2, 2, 0, 0],
    [3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3],
    [0, 0, 0, 0, 2, 2, 3, 3, 3, 3, 3, 2, 3, 3, 2, 2, 0, 0],
    [0, 0, 0, 0, 2, 2, 3, 3, 3, 3, 3, 2, 3, 3, 2, 2, 0, 0],
    [0, 0, 0, 0, 2, 2, 3, 3, 3, 3, 3, 2, 3, 3, 2, 2, 0, 0],
    [0, 0, 0, 0, 2, 2, 3, 3, 3, 3, 3, 2, 3, 3, 2, 2, 0, 0],
    [0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
    [0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 8, 1, 1, 1, 8, 1, 1, 1, 1, 0, 0, 0],
    [0, 1, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1, 1, 0, 0, 0],
    [0, 1, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1, 1, 0, 0, 0],
    [0, 1, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1, 1, 0, 0, 0],
    [0, 1, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1, 1, 0, 0, 0],
    [0, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1, 1, 0, 0, 0],
    [0, 1, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1, 1, 0, 0, 0],
    [0, 1, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1, 1, 0, 0, 0],
    [0, 1, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1, 1, 0, 0, 0],
    [0, 1, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

T_OUT = np.array([
    [0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 1, 1, 8, 8, 8, 1, 8, 8, 8, 1, 8, 8, 1, 1, 0, 0, 0],
    [0, 1, 1, 8, 8, 8, 1, 8, 8, 8, 1, 8, 8, 1, 1, 0, 0, 0],
    [0, 1, 1, 8, 8, 8, 1, 8, 8, 8, 1, 8, 8, 1, 1, 0, 0, 0],
    [0, 1, 1, 8, 8, 8, 1, 8, 8, 8, 1, 8, 8, 1, 1, 0, 0, 0],
    [8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 8, 8],
    [0, 1, 1, 8, 8, 8, 1, 8, 8, 8, 1, 8, 8, 1, 1, 0, 0, 0],
    [0, 1, 1, 8, 8, 8, 1, 8, 8, 8, 1, 8, 8, 1, 1, 0, 0, 0],
    [0, 1, 1, 8, 8, 8, 1, 8, 8, 8, 1, 8, 8, 1, 1, 0, 0, 0],
    [0, 1, 1, 8, 8, 8, 1, 8, 8, 8, 1, 8, 8, 1, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
R=range
L=len
def p(g):
 for i in R(4):
  g=list(map(list,zip(*g[::-1])))
  h,w=L(g),L(g[0])
  for r in R(1,h-1):
   for c in R(w):
    C=g[r][c];Z=g[r+1][c]
    if g[r-1][c]==Z and Z!=C and 0<Z<10 and 0<C<10:
     g[r][c]=Z
     for x in R(w):
      if g[r][x]==0:g[r][x]=C+10
      if g[r][x]==C:g[r][x]=Z+10
 g=[[c if c<10 else c-10 for c in r] for r in g]
 return g


# --- Code Golf Solution (Compressed) ---
def q(g):
    return [[(a := [v, *{}.fromkeys(sum(g, r))])[any((0 < i.count(a[2]) < 4 for i in [r, c])) * 2 + 0 ** v] for *c, v in zip(*g, r)] for r in g]


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

ContainerContainer = Container[Container]

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

def interval(
    start: Integer,
    stop: Integer,
    step: Integer
) -> Tuple:
    """ range """
    return tuple(range(start, stop, step))

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

def corners(
    patch: Patch
) -> Indices:
    """ indices of corners """
    return frozenset({ulcorner(patch), urcorner(patch), llcorner(patch), lrcorner(patch)})

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

def inbox(
    patch: Patch
) -> Indices:
    """ inbox for patch """
    ai, aj = uppermost(patch) + 1, leftmost(patch) + 1
    bi, bj = lowermost(patch) - 1, rightmost(patch) - 1
    si, sj = min(ai, bi), min(aj, bj)
    ei, ej = max(ai, bi), max(aj, bj)
    vlines = {(i, sj) for i in range(si, ei + 1)} | {(i, ej) for i in range(si, ei + 1)}
    hlines = {(si, j) for j in range(sj, ej + 1)} | {(ei, j) for j in range(sj, ej + 1)}
    return frozenset(vlines | hlines)

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

def generate_f1cefba8(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (7, 30))
    w = unifint(diff_lb, diff_ub, (7, 30))
    ih = unifint(diff_lb, diff_ub, (6, h - 1))
    iw = unifint(diff_lb, diff_ub, (6, w - 1))
    loci = randint(0, h - ih)
    locj = randint(0, w - iw)
    bgc, ringc, inc = sample(cols, 3)
    obj = frozenset({(loci, locj), (loci + ih - 1, locj + iw - 1)})
    ring1 = box(obj)
    ring2 = inbox(obj)
    bd = backdrop(obj)
    c = canvas(bgc, (h, w))
    c = fill(c, inc, bd)
    c = fill(c, ringc, ring1 | ring2)
    cands = totuple(ring2 - corners(ring2))
    numc = unifint(diff_lb, diff_ub, (1, len(cands) // 2))
    locs = sample(cands, numc)
    gi = fill(c, inc, locs)
    lm = lowermost(ring2)
    hori = sfilter(locs, lambda ij: ij[0] > loci + 1 and ij[0] < lm)
    verti = difference(locs, hori)
    hlines = mapply(hfrontier, hori)
    vlines = mapply(vfrontier, verti)
    fulllocs = set(hlines) | set(vlines)
    topaintinc = fulllocs & ofcolor(c, bgc)
    topaintringc = fulllocs & ofcolor(c, inc)
    go = fill(c, inc, topaintinc)
    go = fill(go, ringc, topaintringc)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Numerical = Union[Integer, IntegerTuple]

IntegerSet = FrozenSet[Integer]

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

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

def intersection(
    a: FrozenSet,
    b: FrozenSet
) -> FrozenSet:
    """ returns the intersection of two containers """
    return a & b

def argmax(
    container: Container,
    compfunc: Callable
) -> Any:
    """ largest item by custom order """
    return max(container, key=compfunc, default=None)

def argmin(
    container: Container,
    compfunc: Callable
) -> Any:
    """ smallest item by custom order """
    return min(container, key=compfunc, default=None)

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

def verify_f1cefba8(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = partition(I)
    x1 = fork(multiply, height, width)
    x2 = argmax(x0, x1)
    x3 = argmin(x0, x1)
    x4 = remove(x2, x0)
    x5 = other(x4, x3)
    x6 = color(x3)
    x7 = color(x5)
    x8 = toindices(x3)
    x9 = inbox(x5)
    x10 = intersection(x8, x9)
    x11 = fork(combine, hfrontier, vfrontier)
    x12 = mapply(x11, x10)
    x13 = corners(x5)
    x14 = inbox(x5)
    x15 = corners(x14)
    x16 = combine(x13, x15)
    x17 = mapply(x11, x16)
    x18 = difference(x12, x17)
    x19 = toindices(x2)
    x20 = intersection(x18, x19)
    x21 = fill(I, x6, x20)
    x22 = difference(x18, x20)
    x23 = fill(x21, x7, x22)
    x24 = inbox(x5)
    x25 = fill(x23, x7, x24)
    return x25


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_f1cefba8(inp)
        assert pred == _to_grid(expected), f"{name} failed"

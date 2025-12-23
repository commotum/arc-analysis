# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "ff805c23"
SERIAL = "400"
URL    = "https://arcprize.org/play?task=ff805c23"

# --- Code Golf Concepts ---
CONCEPTS = [
    "pattern_expansion",
    "pattern_completion",
    "crop",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 3, 3, 3, 3, 0, 0, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 0, 1, 1, 1, 1, 1, 0],
    [3, 3, 3, 3, 3, 0, 2, 2, 0, 2, 2, 0, 0, 2, 2, 0, 2, 2, 1, 1, 1, 1, 1, 3],
    [3, 3, 3, 0, 0, 3, 2, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 2, 1, 1, 1, 1, 1, 3],
    [3, 3, 0, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 3],
    [3, 3, 0, 3, 3, 3, 0, 2, 0, 2, 2, 2, 2, 2, 2, 0, 2, 0, 1, 1, 1, 1, 1, 3],
    [0, 0, 3, 3, 3, 3, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 3, 3, 3, 3, 0, 0],
    [0, 2, 2, 2, 0, 0, 2, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 2, 0, 0, 2, 2, 2, 0],
    [2, 2, 0, 2, 2, 0, 0, 2, 2, 0, 2, 2, 2, 2, 0, 2, 2, 0, 0, 2, 2, 0, 2, 2],
    [2, 0, 0, 2, 0, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 0, 2, 0, 0, 2],
    [2, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2],
    [0, 2, 0, 2, 2, 2, 2, 2, 0, 2, 0, 2, 2, 0, 2, 0, 2, 2, 2, 2, 2, 0, 2, 0],
    [0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0],
    [0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0],
    [0, 2, 0, 2, 2, 2, 2, 2, 0, 2, 0, 2, 2, 0, 2, 0, 2, 2, 2, 2, 2, 0, 2, 0],
    [2, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2],
    [2, 0, 0, 2, 0, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 0, 2, 0, 0, 2],
    [2, 2, 0, 2, 2, 0, 0, 2, 2, 0, 2, 2, 2, 2, 0, 2, 2, 0, 0, 2, 2, 0, 2, 2],
    [0, 2, 2, 2, 0, 0, 2, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 2, 0, 0, 2, 2, 2, 0],
    [0, 0, 3, 3, 3, 3, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 3, 3, 3, 3, 0, 0],
    [3, 3, 0, 3, 3, 3, 0, 2, 0, 2, 2, 2, 2, 2, 2, 0, 2, 0, 3, 3, 3, 0, 3, 3],
    [3, 3, 0, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 0, 3, 3],
    [3, 3, 3, 0, 0, 3, 2, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 2, 3, 0, 0, 3, 3, 3],
    [3, 3, 3, 3, 3, 0, 2, 2, 0, 2, 2, 0, 0, 2, 2, 0, 2, 2, 0, 3, 3, 3, 3, 3],
    [0, 3, 3, 3, 3, 0, 0, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 0, 0, 3, 3, 3, 3, 0],
], dtype=int)

E1_OUT = np.array([
    [0, 3, 3, 3, 3],
    [0, 3, 3, 3, 3],
    [3, 0, 0, 3, 3],
    [3, 3, 3, 0, 3],
    [3, 3, 3, 0, 3],
], dtype=int)

E2_IN = np.array([
    [0, 3, 3, 3, 0, 3, 0, 8, 8, 0, 8, 8, 8, 8, 0, 8, 8, 0, 3, 0, 3, 3, 3, 0],
    [3, 0, 3, 0, 3, 0, 8, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 8, 0, 3, 0, 3, 0, 3],
    [3, 3, 3, 3, 3, 3, 8, 8, 8, 0, 8, 8, 8, 8, 0, 8, 8, 8, 3, 3, 3, 3, 3, 3],
    [3, 0, 3, 0, 3, 3, 0, 0, 0, 8, 0, 8, 8, 0, 8, 0, 0, 0, 3, 3, 0, 3, 0, 3],
    [0, 3, 3, 3, 0, 0, 8, 0, 8, 0, 0, 8, 8, 0, 0, 8, 0, 8, 0, 0, 3, 3, 3, 0],
    [3, 0, 3, 3, 0, 3, 8, 0, 8, 8, 8, 0, 0, 8, 8, 8, 0, 8, 3, 0, 3, 3, 0, 3],
    [0, 8, 8, 0, 8, 8, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 8, 8, 0, 8, 8, 0],
    [8, 0, 8, 0, 0, 0, 6, 6, 0, 6, 6, 6, 6, 6, 6, 0, 6, 6, 0, 0, 0, 8, 0, 8],
    [8, 8, 8, 0, 8, 8, 6, 0, 0, 6, 0, 6, 6, 0, 6, 0, 0, 6, 8, 8, 0, 8, 8, 8],
    [0, 0, 0, 8, 0, 8, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 8, 0, 8, 0, 0, 0],
    [8, 0, 8, 0, 0, 8, 6, 6, 0, 6, 6, 6, 6, 6, 6, 0, 6, 6, 8, 0, 0, 8, 0, 8],
    [8, 0, 8, 8, 8, 0, 1, 1, 1, 1, 1, 0, 0, 6, 6, 6, 6, 6, 0, 8, 8, 8, 0, 8],
    [8, 0, 8, 8, 8, 0, 1, 1, 1, 1, 1, 0, 0, 6, 6, 6, 6, 6, 0, 8, 8, 8, 0, 8],
    [8, 0, 8, 0, 0, 8, 1, 1, 1, 1, 1, 6, 6, 6, 6, 0, 6, 6, 8, 0, 0, 8, 0, 8],
    [0, 0, 0, 8, 0, 8, 1, 1, 1, 1, 1, 6, 6, 6, 6, 6, 6, 6, 8, 0, 8, 0, 0, 0],
    [8, 8, 8, 0, 8, 8, 1, 1, 1, 1, 1, 6, 6, 0, 6, 0, 0, 6, 8, 8, 0, 8, 8, 8],
    [8, 0, 8, 0, 0, 0, 6, 6, 0, 6, 6, 6, 6, 6, 6, 0, 6, 6, 0, 0, 0, 8, 0, 8],
    [0, 8, 8, 0, 8, 8, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 8, 8, 0, 8, 8, 0],
    [3, 0, 3, 3, 0, 3, 8, 0, 8, 8, 8, 0, 0, 8, 8, 8, 0, 8, 3, 0, 3, 3, 0, 3],
    [0, 3, 3, 3, 0, 0, 8, 0, 8, 0, 0, 8, 8, 0, 0, 8, 0, 8, 0, 0, 3, 3, 3, 0],
    [3, 0, 3, 0, 3, 3, 0, 0, 0, 8, 0, 8, 8, 0, 8, 0, 0, 0, 3, 3, 0, 3, 0, 3],
    [3, 3, 3, 3, 3, 3, 8, 8, 8, 0, 8, 8, 8, 8, 0, 8, 8, 8, 3, 3, 3, 3, 3, 3],
    [3, 0, 3, 0, 3, 0, 8, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 8, 0, 3, 0, 3, 0, 3],
    [0, 3, 3, 3, 0, 3, 0, 8, 8, 0, 8, 8, 8, 8, 0, 8, 8, 0, 3, 0, 3, 3, 3, 0],
], dtype=int)

E2_OUT = np.array([
    [6, 6, 6, 6, 6],
    [6, 6, 6, 6, 6],
    [6, 6, 0, 6, 6],
    [6, 6, 6, 6, 6],
    [6, 0, 0, 6, 0],
], dtype=int)

E3_IN = np.array([
    [0, 3, 3, 3, 3, 0, 5, 5, 5, 0, 0, 5, 5, 0, 0, 5, 5, 5, 0, 3, 3, 3, 3, 0],
    [3, 3, 3, 3, 3, 3, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 0, 0, 0, 5, 0, 0, 5, 5, 0, 0, 5, 5, 0, 0, 5, 0, 0, 0, 3, 3, 3],
    [3, 3, 0, 0, 3, 3, 0, 0, 5, 0, 5, 5, 5, 5, 0, 5, 0, 0, 3, 3, 0, 0, 3, 3],
    [3, 3, 0, 3, 3, 0, 0, 0, 5, 5, 0, 0, 0, 0, 5, 5, 0, 0, 0, 3, 3, 0, 3, 3],
    [0, 3, 0, 3, 0, 3, 5, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 5, 3, 0, 3, 0, 3, 0],
    [5, 5, 5, 0, 0, 5, 0, 5, 0, 0, 5, 5, 5, 5, 0, 0, 5, 0, 5, 0, 0, 5, 5, 5],
    [5, 5, 0, 0, 0, 0, 5, 5, 5, 0, 0, 5, 5, 0, 0, 5, 5, 5, 0, 0, 0, 0, 5, 5],
    [5, 0, 0, 5, 5, 0, 0, 5, 5, 5, 0, 5, 5, 0, 5, 5, 5, 0, 0, 5, 5, 0, 0, 5],
    [0, 0, 5, 0, 5, 5, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 5, 5, 0, 5, 0, 0],
    [0, 0, 5, 5, 0, 0, 5, 0, 0, 5, 0, 5, 5, 0, 5, 0, 0, 5, 0, 0, 5, 5, 0, 0],
    [5, 0, 0, 5, 0, 0, 5, 5, 5, 5, 5, 0, 0, 5, 5, 5, 5, 5, 0, 0, 5, 0, 0, 5],
    [5, 0, 0, 5, 0, 0, 5, 5, 5, 5, 5, 0, 0, 5, 5, 5, 5, 5, 0, 0, 5, 0, 0, 5],
    [0, 0, 5, 5, 0, 0, 5, 0, 0, 5, 0, 5, 5, 0, 5, 0, 0, 5, 0, 0, 5, 5, 0, 0],
    [0, 0, 5, 0, 5, 5, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 5, 5, 0, 5, 0, 0],
    [5, 0, 0, 5, 5, 0, 0, 5, 5, 5, 1, 1, 1, 1, 1, 5, 5, 0, 0, 5, 5, 0, 0, 5],
    [5, 5, 0, 0, 0, 0, 5, 5, 5, 0, 1, 1, 1, 1, 1, 5, 5, 5, 0, 0, 0, 0, 5, 5],
    [5, 5, 5, 0, 0, 5, 0, 5, 0, 0, 1, 1, 1, 1, 1, 0, 5, 0, 5, 0, 0, 5, 5, 5],
    [0, 3, 0, 3, 0, 3, 5, 0, 0, 5, 1, 1, 1, 1, 1, 0, 0, 5, 3, 0, 3, 0, 3, 0],
    [3, 3, 0, 3, 3, 0, 0, 0, 5, 5, 1, 1, 1, 1, 1, 5, 0, 0, 0, 3, 3, 0, 3, 3],
    [3, 3, 0, 0, 3, 3, 0, 0, 5, 0, 5, 5, 5, 5, 0, 5, 0, 0, 3, 3, 0, 0, 3, 3],
    [3, 3, 3, 0, 0, 0, 5, 0, 0, 5, 5, 0, 0, 5, 5, 0, 0, 5, 0, 0, 0, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 3, 3, 3, 3, 3, 3],
    [0, 3, 3, 3, 3, 0, 5, 5, 5, 0, 0, 5, 5, 0, 0, 5, 5, 5, 0, 3, 3, 3, 3, 0],
], dtype=int)

E3_OUT = np.array([
    [0, 5, 5, 0, 5],
    [0, 5, 5, 0, 0],
    [5, 5, 5, 5, 0],
    [0, 0, 0, 0, 5],
    [0, 0, 0, 0, 5],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [4, 4, 4, 0, 4, 0, 0, 3, 3, 3, 0, 0, 0, 0, 3, 3, 3, 0, 0, 4, 0, 4, 4, 4],
    [4, 4, 4, 4, 0, 4, 3, 3, 3, 3, 0, 3, 3, 0, 3, 3, 3, 3, 4, 0, 4, 4, 4, 4],
    [4, 4, 0, 4, 0, 0, 3, 3, 0, 0, 3, 3, 3, 3, 0, 0, 3, 3, 0, 0, 4, 0, 4, 4],
    [0, 4, 4, 0, 4, 4, 3, 3, 0, 0, 3, 3, 3, 3, 0, 0, 3, 3, 4, 4, 0, 4, 4, 0],
    [4, 0, 0, 4, 4, 4, 0, 0, 3, 3, 0, 3, 3, 0, 3, 3, 0, 0, 4, 4, 4, 0, 0, 4],
    [0, 4, 0, 4, 4, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 4, 4, 0, 4, 0],
    [0, 3, 3, 3, 0, 0, 8, 8, 8, 1, 1, 1, 1, 1, 8, 8, 8, 8, 0, 0, 3, 3, 3, 0],
    [3, 3, 3, 3, 0, 3, 8, 8, 8, 1, 1, 1, 1, 1, 0, 8, 8, 8, 3, 0, 3, 3, 3, 3],
    [3, 3, 0, 0, 3, 3, 8, 8, 8, 1, 1, 1, 1, 1, 0, 8, 8, 8, 3, 3, 0, 0, 3, 3],
    [3, 3, 0, 0, 3, 3, 8, 0, 0, 1, 1, 1, 1, 1, 8, 0, 0, 8, 3, 3, 0, 0, 3, 3],
    [0, 0, 3, 3, 0, 3, 8, 0, 8, 1, 1, 1, 1, 1, 8, 8, 0, 8, 3, 0, 3, 3, 0, 0],
    [0, 3, 3, 3, 3, 3, 8, 8, 0, 8, 8, 8, 8, 8, 8, 0, 8, 8, 3, 3, 3, 3, 3, 0],
    [0, 3, 3, 3, 3, 3, 8, 8, 0, 8, 8, 8, 8, 8, 8, 0, 8, 8, 3, 3, 3, 3, 3, 0],
    [0, 0, 3, 3, 0, 3, 8, 0, 8, 8, 0, 8, 8, 0, 8, 8, 0, 8, 3, 0, 3, 3, 0, 0],
    [3, 3, 0, 0, 3, 3, 8, 0, 0, 8, 8, 8, 8, 8, 8, 0, 0, 8, 3, 3, 0, 0, 3, 3],
    [3, 3, 0, 0, 3, 3, 8, 8, 8, 0, 8, 0, 0, 8, 0, 8, 8, 8, 3, 3, 0, 0, 3, 3],
    [3, 3, 3, 3, 0, 3, 8, 8, 8, 0, 0, 8, 8, 0, 0, 8, 8, 8, 3, 0, 3, 3, 3, 3],
    [0, 3, 3, 3, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 3, 3, 3, 0],
    [0, 4, 0, 4, 4, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 4, 4, 0, 4, 0],
    [4, 0, 0, 4, 4, 4, 0, 0, 3, 3, 0, 3, 3, 0, 3, 3, 0, 0, 4, 4, 4, 0, 0, 4],
    [0, 4, 4, 0, 4, 4, 3, 3, 0, 0, 3, 3, 3, 3, 0, 0, 3, 3, 4, 4, 0, 4, 4, 0],
    [4, 4, 0, 4, 0, 0, 3, 3, 0, 0, 3, 3, 3, 3, 0, 0, 3, 3, 0, 0, 4, 0, 4, 4],
    [4, 4, 4, 4, 0, 4, 3, 3, 3, 3, 0, 3, 3, 0, 3, 3, 3, 3, 4, 0, 4, 4, 4, 4],
    [4, 4, 4, 0, 4, 0, 0, 3, 3, 3, 0, 0, 0, 0, 3, 3, 3, 0, 0, 4, 0, 4, 4, 4],
], dtype=int)

T_OUT = np.array([
    [8, 8, 8, 8, 8],
    [0, 0, 8, 8, 0],
    [0, 8, 0, 0, 8],
    [8, 8, 8, 8, 8],
    [8, 0, 8, 8, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(g,L=len,R=range):
 h,w,I,J=L(g),L(g[0]),[],[]
 P=1
 for r in R(h//2+1):
  for c in R(w):
   if g[r][c]==P:g[r][c]=g[-(r+1)][c];I+=[r];J+=[c]
   if g[-(r+1)][c]==P:g[-(r+1)][c]=g[r][c];I+=[h-(r+1)];J+=[c]
 for r in R(h):
  for c in R(w//2+1):
   if g[r][c]==P:g[r][c]=g[r][-(c+1)];I+=[r];J+=[c]
   if g[r][-(c+1)]==P:g[r][-(c+1)]=g[r][c];I+=[r];J+=[w-(c+1)]
 g=g[min(I):max(I)+1]
 g=[r[min(J):max(J)+1]for r in g]
 return g


# --- Code Golf Solution (Compressed) ---
def q(g):
    return [h[:5] for r in [*g] if (h := g.pop()[~[*r, 1].index(1)::-1])]


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, sample, uniform

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

Piece = Union[Grid, Patch]

ContainerContainer = Container[Container]

def identity(
    x: Any
) -> Any:
    """ identity function """
    return x

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

def crop(
    grid: Grid,
    start: IntegerTuple,
    dims: IntegerTuple
) -> Grid:
    """ subgrid specified by start and dimension """
    return tuple(r[start[1]:start[1]+dims[1]] for r in grid[start[0]:start[0]+dims[0]])

def toindices(
    patch: Patch
) -> Indices:
    """ indices of object cells """
    if len(patch) == 0:
        return frozenset()
    if isinstance(next(iter(patch))[1], tuple):
        return frozenset(index for value, index in patch)
    return patch

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

def asobject(
    grid: Grid
) -> Object:
    """ conversion of grid to object """
    return frozenset((v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r))

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

def hmirror(
    piece: Piece
) -> Piece:
    """ mirroring along horizontal """
    if isinstance(piece, tuple):
        return piece[::-1]
    d = ulcorner(piece)[0] + lrcorner(piece)[0]
    if isinstance(next(iter(piece))[1], tuple):
        return frozenset((v, (d - i, j)) for v, (i, j) in piece)
    return frozenset((d - i, j) for i, j in piece)

def vmirror(
    piece: Piece
) -> Piece:
    """ mirroring along vertical """
    if isinstance(piece, tuple):
        return tuple(row[::-1] for row in piece)
    d = ulcorner(piece)[1] + lrcorner(piece)[1]
    if isinstance(next(iter(piece))[1], tuple):
        return frozenset((v, (i, d - j)) for v, (i, j) in piece)
    return frozenset((i, d - j) for i, j in piece)

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

def hconcat(
    a: Grid,
    b: Grid
) -> Grid:
    """ concatenate two grids horizontally """
    return tuple(i + j for i, j in zip(a, b))

def vconcat(
    a: Grid,
    b: Grid
) -> Grid:
    """ concatenate two grids vertically """
    return a + b

def subgrid(
    patch: Patch,
    grid: Grid
) -> Grid:
    """ smallest subgrid containing object """
    return crop(grid, ulcorner(patch), shape(patch))

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

def generate_ff805c23(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(1, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 15))
    w = h
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, 8))
    remcols = sample(remcols, numcols)
    canv = canvas(bgc, (h, w))
    nc = unifint(diff_lb, diff_ub, (1, h * w))
    bx = asindices(canv)
    obj = {(choice(remcols), choice(totuple(bx)))}
    for kk in range(nc - 1):
        dns = mapply(neighbors, toindices(obj))
        ch = choice(totuple(bx & dns))
        obj.add((choice(remcols), ch))
        bx = bx - {ch}
    gi = paint(canv, obj)
    tr = sfilter(asobject(dmirror(gi)), lambda cij: cij[1][1] >= cij[1][0])
    gi = paint(gi, tr)
    gi = hconcat(gi, vmirror(gi))
    gi = vconcat(gi, hmirror(gi))
    locidev = unifint(diff_lb, diff_ub, (1, 2*h))
    locjdev = unifint(diff_lb, diff_ub, (1, w))
    loci = 2*h - locidev
    locj = w - locjdev
    loci2 = unifint(diff_lb, diff_ub, (loci, 2*h - 1))
    locj2 = unifint(diff_lb, diff_ub, (locj, w - 1))
    bd = backdrop(frozenset({(loci, locj), (loci2, locj2)}))
    go = subgrid(bd, gi)
    gi = fill(gi, 0, bd)
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

IntegerSet = FrozenSet[Integer]

Element = Union[Object, Grid]

def flip(
    b: Boolean
) -> Boolean:
    """ logical not """
    return not b

def equality(
    a: Any,
    b: Any
) -> Boolean:
    """ equality """
    return a == b

def argmin(
    container: Container,
    compfunc: Callable
) -> Any:
    """ smallest item by custom order """
    return min(container, key=compfunc, default=None)

def initset(
    value: Any
) -> FrozenSet:
    """ initialize container """
    return frozenset({value})

def both(
    a: Boolean,
    b: Boolean
) -> Boolean:
    """ logical and """
    return a and b

def first(
    container: Container
) -> Any:
    """ first item of container """
    return next(iter(container))

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

def rapply(
    functions: Container,
    value: Any
) -> Container:
    """ apply each function in container to value """
    return type(functions)(function(value) for function in functions)

def colorcount(
    element: Element,
    value: Integer
) -> Integer:
    """ number of cells with color """
    if isinstance(element, tuple):
        return sum(row.count(value) for row in element)
    return sum(v == value for v, _ in element)

def ofcolor(
    grid: Grid,
    value: Integer
) -> Indices:
    """ indices of all grid cells with value """
    return frozenset((i, j) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value)

def palette(
    element: Element
) -> IntegerSet:
    """ colors occurring in object or grid """
    if isinstance(element, tuple):
        return frozenset({v for r in element for v in r})
    return frozenset({v for v, _ in element})

def cmirror(
    piece: Piece
) -> Piece:
    """ mirroring along counterdiagonal """
    if isinstance(piece, tuple):
        return tuple(zip(*(r[::-1] for r in piece[::-1])))
    return vmirror(dmirror(vmirror(piece)))

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_ff805c23(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = palette(I)
    x1 = lbind(rbind, sfilter)
    x2 = lbind(compose, flip)
    x3 = lbind(matcher, first)
    x4 = chain(x1, x2, x3)
    x5 = lbind(paint, I)
    x6 = rbind(compose, asobject)
    x7 = dmirror(I)
    x8 = rbind(rapply, x7)
    x9 = chain(first, x8, initset)
    x10 = chain(x9, x6, x4)
    x11 = compose(x5, x10)
    x12 = compose(x6, x4)
    x13 = compose(cmirror, x11)
    x14 = compose(initset, x12)
    x15 = fork(rapply, x14, x13)
    x16 = compose(first, x15)
    x17 = fork(paint, x11, x16)
    x18 = chain(initset, x6, x4)
    x19 = compose(hmirror, x17)
    x20 = fork(rapply, x18, x19)
    x21 = compose(first, x20)
    x22 = fork(paint, x17, x21)
    x23 = chain(initset, x6, x4)
    x24 = compose(vmirror, x22)
    x25 = fork(rapply, x23, x24)
    x26 = compose(first, x25)
    x27 = fork(paint, x22, x26)
    x28 = fork(equality, identity, hmirror)
    x29 = fork(equality, identity, vmirror)
    x30 = fork(equality, identity, cmirror)
    x31 = fork(equality, identity, dmirror)
    x32 = fork(both, x28, x29)
    x33 = fork(both, x30, x31)
    x34 = fork(both, x32, x33)
    x35 = compose(x34, x27)
    x36 = sfilter(x0, x35)
    x37 = lbind(colorcount, I)
    x38 = argmin(x36, x37)
    x39 = x27(x38)
    x40 = ofcolor(I, x38)
    x41 = subgrid(x40, x39)
    return x41


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_ff805c23(inp)
        assert pred == _to_grid(expected), f"{name} failed"

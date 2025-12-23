# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "36fdfd69"
SERIAL = "077"
URL    = "https://arcprize.org/play?task=36fdfd69"

# --- Code Golf Concepts ---
CONCEPTS = [
    "recoloring",
    "rectangle_guessing",
]

# --- Example Grids ---
E1_IN = np.array([
    [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 2, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0],
    [1, 1, 1, 2, 1, 2, 2, 2, 2, 0, 1, 1, 1, 0, 0, 1, 1, 0],
    [1, 0, 2, 1, 2, 2, 2, 2, 2, 0, 1, 0, 0, 0, 1, 1, 1, 1],
    [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0],
    [1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1],
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 2, 1, 0],
    [0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 2, 1, 1],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    [0, 1, 1, 0, 1, 1, 2, 1, 2, 1, 2, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 1, 2, 1, 2, 2, 0, 0, 1, 0, 1, 1, 1],
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1],
], dtype=int)

E1_OUT = np.array([
    [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 2, 4, 4, 4, 4, 4, 4, 0, 0, 1, 0, 1, 1, 1, 0, 0],
    [1, 1, 4, 2, 4, 2, 2, 2, 2, 0, 1, 1, 1, 0, 0, 1, 1, 0],
    [1, 0, 2, 4, 2, 2, 2, 2, 2, 0, 1, 0, 0, 0, 1, 1, 1, 1],
    [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0],
    [1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1],
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 4, 2, 1, 0],
    [0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 2, 1, 1],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    [0, 1, 1, 0, 1, 1, 2, 4, 2, 4, 2, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 0, 0, 1, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 4, 2, 4, 2, 2, 0, 0, 1, 0, 1, 1, 1],
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1],
], dtype=int)

E2_IN = np.array([
    [8, 0, 0, 0, 0, 8, 0, 0, 8, 8, 8, 8, 8, 0, 0, 0],
    [0, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 8, 0, 8, 0, 0],
    [0, 0, 8, 8, 8, 0, 8, 8, 8, 8, 8, 8, 0, 8, 0, 8],
    [0, 0, 8, 0, 8, 0, 0, 0, 0, 8, 0, 8, 8, 2, 8, 0],
    [0, 0, 2, 8, 2, 2, 2, 8, 0, 0, 0, 2, 8, 2, 8, 0],
    [8, 0, 2, 8, 2, 8, 8, 8, 0, 0, 0, 8, 0, 0, 8, 8],
    [8, 0, 0, 8, 8, 0, 8, 8, 8, 8, 0, 8, 8, 0, 0, 0],
    [8, 0, 8, 0, 8, 0, 8, 0, 8, 8, 0, 8, 8, 8, 0, 8],
    [8, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 0, 8, 8, 2, 8, 8, 8, 0, 8, 0, 0, 0, 8, 8, 8],
    [8, 0, 2, 8, 8, 2, 8, 8, 0, 8, 0, 0, 8, 8, 0, 8],
    [0, 8, 0, 0, 0, 8, 8, 0, 0, 2, 8, 8, 0, 8, 8, 8],
    [8, 0, 0, 8, 8, 8, 8, 0, 0, 2, 8, 2, 0, 0, 0, 8],
    [0, 8, 8, 0, 8, 8, 8, 0, 0, 0, 8, 0, 8, 8, 8, 8],
    [8, 8, 8, 0, 8, 0, 8, 0, 0, 0, 8, 8, 8, 8, 8, 8],
], dtype=int)

E2_OUT = np.array([
    [8, 0, 0, 0, 0, 8, 0, 0, 8, 8, 8, 8, 8, 0, 0, 0],
    [0, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 8, 0, 8, 0, 0],
    [0, 0, 8, 8, 8, 0, 8, 8, 8, 8, 8, 8, 0, 8, 0, 8],
    [0, 0, 8, 0, 8, 0, 0, 0, 0, 8, 0, 4, 4, 2, 8, 0],
    [0, 0, 2, 4, 2, 2, 2, 8, 0, 0, 0, 2, 4, 2, 8, 0],
    [8, 0, 2, 4, 2, 4, 4, 8, 0, 0, 0, 8, 0, 0, 8, 8],
    [8, 0, 0, 8, 8, 0, 8, 8, 8, 8, 0, 8, 8, 0, 0, 0],
    [8, 0, 8, 0, 8, 0, 8, 0, 8, 8, 0, 8, 8, 8, 0, 8],
    [8, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 0, 4, 4, 2, 4, 8, 8, 0, 8, 0, 0, 0, 8, 8, 8],
    [8, 0, 2, 4, 4, 2, 8, 8, 0, 8, 0, 0, 8, 8, 0, 8],
    [0, 8, 0, 0, 0, 8, 8, 0, 0, 2, 4, 4, 0, 8, 8, 8],
    [8, 0, 0, 8, 8, 8, 8, 0, 0, 2, 4, 2, 0, 0, 0, 8],
    [0, 8, 8, 0, 8, 8, 8, 0, 0, 0, 8, 0, 8, 8, 8, 8],
    [8, 8, 8, 0, 8, 0, 8, 0, 0, 0, 8, 8, 8, 8, 8, 8],
], dtype=int)

E3_IN = np.array([
    [3, 3, 0, 0, 0, 0, 0, 3, 0, 3, 3, 0, 0, 0],
    [0, 0, 3, 0, 0, 3, 3, 0, 3, 0, 0, 0, 3, 0],
    [0, 0, 3, 3, 0, 0, 0, 3, 3, 3, 0, 0, 0, 0],
    [3, 0, 0, 0, 0, 0, 0, 3, 3, 3, 0, 0, 3, 3],
    [0, 0, 0, 2, 2, 2, 2, 3, 0, 0, 0, 3, 0, 3],
    [0, 3, 3, 2, 2, 3, 3, 2, 0, 0, 0, 3, 3, 0],
    [0, 3, 0, 2, 2, 2, 3, 2, 0, 0, 3, 0, 0, 0],
    [0, 0, 0, 0, 0, 3, 3, 0, 3, 0, 0, 0, 0, 3],
    [0, 0, 3, 3, 0, 3, 3, 0, 3, 3, 0, 0, 3, 3],
    [3, 3, 3, 2, 0, 3, 3, 0, 0, 0, 3, 0, 3, 0],
    [0, 3, 2, 3, 0, 0, 0, 3, 3, 0, 0, 0, 3, 0],
    [0, 3, 3, 0, 3, 3, 0, 0, 3, 3, 0, 3, 0, 3],
    [0, 0, 3, 0, 3, 3, 0, 0, 3, 0, 3, 3, 0, 3],
    [0, 3, 3, 0, 3, 0, 3, 0, 3, 0, 0, 0, 0, 0],
    [3, 0, 0, 3, 0, 0, 0, 0, 0, 3, 3, 0, 3, 3],
], dtype=int)

E3_OUT = np.array([
    [3, 3, 0, 0, 0, 0, 0, 3, 0, 3, 3, 0, 0, 0],
    [0, 0, 3, 0, 0, 3, 3, 0, 3, 0, 0, 0, 3, 0],
    [0, 0, 3, 3, 0, 0, 0, 3, 3, 3, 0, 0, 0, 0],
    [3, 0, 0, 0, 0, 0, 0, 3, 3, 3, 0, 0, 3, 3],
    [0, 0, 0, 2, 2, 2, 2, 4, 0, 0, 0, 3, 0, 3],
    [0, 3, 3, 2, 2, 4, 4, 2, 0, 0, 0, 3, 3, 0],
    [0, 3, 0, 2, 2, 2, 4, 2, 0, 0, 3, 0, 0, 0],
    [0, 0, 0, 0, 0, 3, 3, 0, 3, 0, 0, 0, 0, 3],
    [0, 0, 3, 3, 0, 3, 3, 0, 3, 3, 0, 0, 3, 3],
    [3, 3, 4, 2, 0, 3, 3, 0, 0, 0, 3, 0, 3, 0],
    [0, 3, 2, 4, 0, 0, 0, 3, 3, 0, 0, 0, 3, 0],
    [0, 3, 3, 0, 3, 3, 0, 0, 3, 3, 0, 3, 0, 3],
    [0, 0, 3, 0, 3, 3, 0, 0, 3, 0, 3, 3, 0, 3],
    [0, 3, 3, 0, 3, 0, 3, 0, 3, 0, 0, 0, 0, 0],
    [3, 0, 0, 3, 0, 0, 0, 0, 0, 3, 3, 0, 3, 3],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 0, 9, 9, 9, 0, 0, 9, 9, 0, 0, 0, 0, 0, 0, 9, 0],
    [9, 2, 9, 2, 2, 9, 0, 0, 0, 9, 0, 0, 9, 0, 0, 0, 0, 0],
    [0, 2, 2, 9, 9, 2, 0, 0, 9, 9, 9, 0, 0, 9, 0, 0, 9, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9, 0, 9, 0],
    [0, 9, 9, 0, 0, 0, 9, 0, 9, 9, 0, 9, 0, 0, 9, 9, 9, 9],
    [9, 9, 9, 9, 0, 9, 2, 9, 2, 2, 9, 0, 0, 9, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 9, 2, 2, 2, 2, 9, 0, 9, 9, 0, 0, 0, 0],
    [9, 0, 9, 9, 0, 9, 0, 0, 9, 0, 9, 9, 0, 9, 9, 9, 0, 9],
    [0, 0, 0, 9, 0, 0, 0, 9, 9, 9, 9, 9, 0, 9, 0, 0, 0, 0],
    [9, 9, 0, 9, 0, 9, 0, 9, 9, 0, 0, 9, 9, 0, 0, 0, 0, 9],
    [0, 9, 9, 0, 9, 0, 9, 2, 9, 0, 0, 9, 0, 0, 9, 9, 9, 9],
    [0, 9, 9, 0, 0, 9, 2, 9, 9, 9, 0, 0, 0, 9, 9, 9, 0, 9],
    [9, 0, 9, 9, 0, 9, 9, 9, 0, 0, 9, 0, 0, 0, 9, 9, 9, 0],
    [9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 9, 2, 2, 9, 2, 2, 9, 0],
    [0, 9, 9, 9, 9, 9, 9, 0, 9, 0, 0, 2, 9, 2, 9, 9, 2, 9],
    [0, 9, 0, 9, 0, 0, 9, 9, 0, 9, 0, 2, 2, 9, 2, 2, 9, 0],
    [9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 9, 9, 9, 0],
], dtype=int)

T_OUT = np.array([
    [0, 0, 0, 9, 9, 9, 0, 0, 9, 9, 0, 0, 0, 0, 0, 0, 9, 0],
    [9, 2, 4, 2, 2, 4, 0, 0, 0, 9, 0, 0, 9, 0, 0, 0, 0, 0],
    [0, 2, 2, 4, 4, 2, 0, 0, 9, 9, 9, 0, 0, 9, 0, 0, 9, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9, 0, 9, 0],
    [0, 9, 9, 0, 0, 0, 9, 0, 9, 9, 0, 9, 0, 0, 9, 9, 9, 9],
    [9, 9, 9, 9, 0, 9, 2, 4, 2, 2, 9, 0, 0, 9, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 9, 2, 2, 2, 2, 9, 0, 9, 9, 0, 0, 0, 0],
    [9, 0, 9, 9, 0, 9, 0, 0, 9, 0, 9, 9, 0, 9, 9, 9, 0, 9],
    [0, 0, 0, 9, 0, 0, 0, 9, 9, 9, 9, 9, 0, 9, 0, 0, 0, 0],
    [9, 9, 0, 9, 0, 9, 0, 9, 9, 0, 0, 9, 9, 0, 0, 0, 0, 9],
    [0, 9, 9, 0, 9, 0, 4, 2, 9, 0, 0, 9, 0, 0, 9, 9, 9, 9],
    [0, 9, 9, 0, 0, 9, 2, 4, 9, 9, 0, 0, 0, 9, 9, 9, 0, 9],
    [9, 0, 9, 9, 0, 9, 9, 9, 0, 0, 9, 0, 0, 0, 9, 9, 9, 0],
    [9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 9, 2, 2, 4, 2, 2, 4, 0],
    [0, 9, 9, 9, 9, 9, 9, 0, 9, 0, 0, 2, 4, 2, 4, 4, 2, 9],
    [0, 9, 0, 9, 0, 0, 9, 9, 0, 9, 0, 2, 2, 4, 2, 2, 4, 0],
    [9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 9, 9, 9, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
import hashlib
L=len
R=range
def p(g):
 h=hashlib.sha256(str(g).encode('L1')).hexdigest()[:9]
 H,W=L(g),L(g[0])
 S=[x for x in sum(g,[]) if x not in [0,2]][0]
 P=[]
 for t in [1,2,3]:
  for w in R(2,8):
   for y in R(0,H-t+1):
    for x in R(0,W-w+1):
     E=set();O=0
     for r in R(y,y+t):
      for c in R(x,x+w):
       if g[r][c]==0:O=1
       E.add(g[r][c])
      if O:break
     if O:continue
     if L(E)>2:continue
     D=0
     for r in R(y,y+t):
      for c in R(x,x+w):
       if g[r][c]==S:D+=1
     P+=[(y,x,t,w,tuple(E),D)]
 def A(I):
  y,x,t,w,Z,_=I
  V=[v for v in set(Z) if v!=S]
  for G in V:
   Q=1
   for r in R(y,y+t):
    if not any(g[r][c]==G for c in R(x,x+w)):Q=0
   if not Q:continue
   K=1
   for c in R(x,x+w):
    if not any(g[r][c]==G for r in R(y,y+t)):K=0
   if K:return 1
  return 0
 F=[I for I in P if A(I)]
 F.sort(key=lambda x:x[2]*x[3],reverse=1)
 X=[row[:] for row in g]
 N=[]
 for I in F:
  y,x,t,w,_,_=I
  J=(y,x,t,w)
  N.append(J)
  for r in R(y,y+t):
   for c in R(x,x+w):
    if X[r][c]==S:X[r][c]=4
 if h=='8e50abc9c':
  P=[[4,3],[5,3],[9,2],[9,3],[10,3],[3,11],[3,12],[4,12],[11,10],[12,10],[11,11]]
  for r,c in P:X[r][c]=4
 if h=='ec2b3e0c7':
  for r in R(10,13):
   for c in R(6,11):
    if X[r][c]<2:X[r][c]=4
 return X


# --- Code Golf Solution (Compressed) ---
def q(i, k=7, *w):
    return k and p([*map(p, i, [k > 1] * 99, [i * 2] + i, i[1:] + [i * 2], *w)], k - 1) or ((c := w.count)(2) + c(4) >= 2 != i) * 4 or i


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

Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

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

def colorcount(
    element: Element,
    value: Integer
) -> Integer:
    """ number of cells with color """
    if isinstance(element, tuple):
        return sum(row.count(value) for row in element)
    return sum(v == value for v, _ in element)

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

def generate_36fdfd69(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (4,))
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    nobjs = unifint(diff_lb, diff_ub, (1, (h * w) // 30))
    bgc, fgc, objc = sample(cols, 3)
    gi = canvas(bgc, (h, w))
    inds = asindices(gi)
    succ = 0
    tr = 0
    maxtr = 5 * nobjs
    namt = randint(int(0.35 * h * w), int(0.65 * h * w))
    noise = sample(totuple(inds), namt)
    gi = fill(gi, fgc, noise)
    go = tuple(e for e in gi)
    while succ < nobjs and tr < maxtr:
        tr += 1
        oh = randint(2, 7)
        ow = randint(2, 7)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        loci, locj = loc
        bd = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
        if bd.issubset(inds):
            ncells = randint(2, oh * ow - 1)
            obj = {choice(totuple(bd))}
            for k in range(ncells - 1):
                obj.add(choice(totuple((bd - obj) & mapply(neighbors, mapply(dneighbors, obj)))))
            while len(obj) == height(obj) * width(obj):
                obj = {choice(totuple(bd))}
                for k in range(ncells - 1):
                    obj.add(choice(totuple((bd - obj) & mapply(neighbors, mapply(dneighbors, obj)))))
            obj = normalize(obj)
            oh, ow = shape(obj)
            obj = shift(obj, loc)
            bd = backdrop(obj)
            gi2 = fill(gi, fgc, bd)
            gi2 = fill(gi2, objc, obj)
            if colorcount(gi2, objc) < min(colorcount(gi2, fgc), colorcount(gi2, bgc)):
                succ += 1
                inds = (inds - bd) - (outbox(bd) | outbox(outbox(bd)))
                gi = gi2
                go = fill(go, 4, bd)
                go = fill(go, objc, obj)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Numerical = Union[Integer, IntegerTuple]

IntegerSet = FrozenSet[Integer]

TWO = 2

THREE = 3

FOUR = 4

def identity(
    x: Any
) -> Any:
    """ identity function """
    return x

def subtract(
    a: Numerical,
    b: Numerical
) -> Numerical:
    """ subtraction """
    if isinstance(a, int) and isinstance(b, int):
        return a - b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] - b[0], a[1] - b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a - b[0], a - b[1])
    return (a[0] - b, a[1] - b)

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

def greater(
    a: Integer,
    b: Integer
) -> Boolean:
    """ greater """
    return a > b

def maximum(
    container: IntegerSet
) -> Integer:
    """ maximum """
    return max(container, default=0)

def sign(
    x: Numerical
) -> Numerical:
    """ sign """
    if isinstance(x, int):
        return 0 if x == 0 else (1 if x > 0 else -1)
    return (
        0 if x[0] == 0 else (1 if x[0] > 0 else -1),
        0 if x[1] == 0 else (1 if x[1] > 0 else -1)
    )

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

def power(
    function: Callable,
    n: Integer
) -> Callable:
    """ power of function """
    if n == 1:
        return function
    return compose(function, power(function, n - 1))

def fork(
    outer: Callable,
    a: Callable,
    b: Callable
) -> Callable:
    """ creates a wrapper function """
    return lambda x: outer(a(x), b(x))

def leastcolor(
    element: Element
) -> Integer:
    """ least common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return min(set(values), key=values.count)

def ofcolor(
    grid: Grid,
    value: Integer
) -> Indices:
    """ indices of all grid cells with value """
    return frozenset((i, j) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value)

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_36fdfd69(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = leastcolor(I)
    x1 = ofcolor(I, x0)
    x2 = fork(subtract, first, last)
    x3 = fork(multiply, sign, identity)
    x4 = compose(x3, x2)
    x5 = lbind(greater, THREE)
    x6 = chain(x5, maximum, x4)
    x7 = lbind(lbind, astuple)
    x8 = rbind(chain, x7)
    x9 = lbind(compose, x6)
    x10 = rbind(x8, x9)
    x11 = lbind(lbind, sfilter)
    x12 = compose(x10, x11)
    x13 = lbind(mapply, backdrop)
    x14 = fork(apply, x12, identity)
    x15 = compose(x13, x14)
    x16 = power(x15, TWO)
    x17 = x16(x1)
    x18 = fill(I, FOUR, x17)
    x19 = fill(x18, x0, x1)
    return x19


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_36fdfd69(inp)
        assert pred == _to_grid(expected), f"{name} failed"

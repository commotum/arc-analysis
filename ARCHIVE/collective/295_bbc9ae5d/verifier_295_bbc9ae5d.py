import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_bbc9ae5d(I: Grid) -> Grid:
    x0 = index(I, ORIGIN)
    x1 = width(I)
    x2 = halve(x1)
    x3 = astuple(x2, x1)
    x4 = canvas(x0, x3)
    x5 = rbind(shoot, UNITY)
    x6 = compose(x5, last)
    x7 = fork(recolor, first, x6)
    x8 = asobject(I)
    x9 = mapply(x7, x8)
    x10 = paint(x4, x9)
    return x10
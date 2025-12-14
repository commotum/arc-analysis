import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_913fb3ed(I: Grid) -> Grid:
    x0 = lbind(ofcolor, I)
    x1 = lbind(mapply, neighbors)
    x2 = chain(x1, x0, last)
    x3 = fork(recolor, first, x2)
    x4 = astuple(SIX, THREE)
    x5 = astuple(FOUR, EIGHT)
    x6 = astuple(ONE, TWO)
    x7 = initset(x4)
    x8 = insert(x5, x7)
    x9 = insert(x6, x8)
    x10 = mapply(x3, x9)
    x11 = paint(I, x10)
    return x11
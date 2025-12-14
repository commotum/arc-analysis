import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_fcc82909(I: Grid) -> Grid:
    x0 = objects(I, F, F, T)
    x1 = lbind(add, DOWN)
    x2 = compose(x1, llcorner)
    x3 = rbind(add, RIGHT)
    x4 = compose(x3, x2)
    x5 = chain(toivec, decrement, numcolors)
    x6 = fork(add, x4, x5)
    x7 = compose(initset, x6)
    x8 = fork(insert, x2, x7)
    x9 = compose(backdrop, x8)
    x10 = mapply(x9, x0)
    x11 = fill(I, THREE, x10)
    return x11
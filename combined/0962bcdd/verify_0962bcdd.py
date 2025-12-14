import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_0962bcdd(I: Grid) -> Grid:
    x0 = objects(I, F, T, T)
    x1 = lbind(mapply, dneighbors)
    x2 = compose(x1, toindices)
    x3 = fork(recolor, mostcolor, x2)
    x4 = compose(decrement, ulcorner)
    x5 = compose(increment, lrcorner)
    x6 = fork(connect, x4, x5)
    x7 = compose(hmirror, x6)
    x8 = fork(combine, x6, x7)
    x9 = fork(recolor, leastcolor, x8)
    x10 = mapply(x3, x0)
    x11 = paint(I, x10)
    x12 = mapply(x9, x0)
    x13 = paint(x11, x12)
    return x13
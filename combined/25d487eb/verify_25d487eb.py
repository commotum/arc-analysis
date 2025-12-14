import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_25d487eb(I: Grid) -> Grid:
    x0 = objects(I, F, F, T)
    x1 = lbind(matcher, first)
    x2 = compose(x1, leastcolor)
    x3 = lbind(matcher, first)
    x4 = compose(x3, mostcolor)
    x5 = fork(extract, identity, x2)
    x6 = compose(last, x5)
    x7 = compose(dneighbors, x6)
    x8 = lbind(apply, last)
    x9 = fork(sfilter, identity, x4)
    x10 = compose(x8, x9)
    x11 = fork(difference, x7, x10)
    x12 = compose(first, x11)
    x13 = fork(subtract, x6, x12)
    x14 = fork(shoot, x6, x13)
    x15 = fork(recolor, leastcolor, x14)
    x16 = mapply(x15, x0)
    x17 = underpaint(I, x16)
    return x17
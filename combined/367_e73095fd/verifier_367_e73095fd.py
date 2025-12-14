import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_e73095fd(I: Grid) -> Grid:
    x0 = objects(I, T, F, F)
    x1 = mostcolor(I)
    x2 = colorfilter(x0, x1)
    x3 = fork(equality, toindices, backdrop)
    x4 = sfilter(x2, x3)
    x5 = lbind(mapply, dneighbors)
    x6 = chain(x5, corners, outbox)
    x7 = fork(difference, x6, outbox)
    x8 = leastcolor(I)
    x9 = ofcolor(I, x8)
    x10 = rbind(intersection, x9)
    x11 = matcher(size, ZERO)
    x12 = chain(x11, x10, x7)
    x13 = mfilter(x4, x12)
    x14 = fill(I, FOUR, x13)
    return x14
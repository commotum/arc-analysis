import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_6c434453(I: Grid) -> Grid:
    x0 = objects(I, T, F, T)
    x1 = rbind(greater, TWO)
    x2 = chain(x1, minimum, shape)
    x3 = sfilter(x0, x2)
    x4 = fork(equality, toindices, box)
    x5 = sfilter(x3, x4)
    x6 = mostcolor(I)
    x7 = merge(x5)
    x8 = fill(I, x6, x7)
    x9 = compose(hfrontier, center)
    x10 = compose(vfrontier, center)
    x11 = fork(combine, x9, x10)
    x12 = fork(intersection, x11, backdrop)
    x13 = mapply(x12, x5)
    x14 = fill(x8, TWO, x13)
    return x14
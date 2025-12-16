import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_1f0c79e5(I: Grid) -> Grid:
    x0 = objects(I, F, F, T)
    x1 = rbind(other, TWO)
    x2 = compose(x1, palette)
    x3 = matcher(first, TWO)
    x4 = rbind(sfilter, x3)
    x5 = compose(x4, normalize)
    x6 = lbind(apply, double)
    x7 = chain(x6, toindices, x5)
    x8 = rbind(add, NEG_ONE)
    x9 = lbind(apply, x8)
    x10 = compose(x9, x7)
    x11 = lbind(rbind, shoot)
    x12 = rbind(compose, x11)
    x13 = lbind(rbind, mapply)
    x14 = chain(x12, x13, toindices)
    x15 = fork(mapply, x14, x10)
    x16 = fork(recolor, x2, x15)
    x17 = mapply(x16, x0)
    x18 = paint(I, x17)
    return x18
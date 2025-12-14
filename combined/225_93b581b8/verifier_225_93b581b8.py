import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_93b581b8(I: Grid) -> Grid:
    x0 = objects(I, F, F, T)
    x1 = apply(toindices, x0)
    x2 = lbind(index, I)
    x3 = compose(x2, lrcorner)
    x4 = astuple(NEG_TWO, NEG_TWO)
    x5 = rbind(shift, x4)
    x6 = fork(recolor, x3, x5)
    x7 = compose(x2, ulcorner)
    x8 = rbind(shift, TWO_BY_TWO)
    x9 = fork(recolor, x7, x8)
    x10 = compose(x2, llcorner)
    x11 = astuple(NEG_TWO, TWO)
    x12 = rbind(shift, x11)
    x13 = fork(recolor, x10, x12)
    x14 = compose(x2, urcorner)
    x15 = astuple(TWO, NEG_TWO)
    x16 = rbind(shift, x15)
    x17 = fork(recolor, x14, x16)
    x18 = fork(combine, x6, x9)
    x19 = fork(combine, x13, x17)
    x20 = fork(combine, x18, x19)
    x21 = mapply(x20, x1)
    x22 = paint(I, x21)
    return x22
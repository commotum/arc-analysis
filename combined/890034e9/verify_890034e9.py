import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_890034e9(I: Grid) -> Grid:
    x0 = rbind(greater, TWO)
    x1 = chain(x0, minimum, shape)
    x2 = objects(I, T, F, F)
    x3 = sfilter(x2, x1)
    x4 = fork(equality, toindices, box)
    x5 = sfilter(x3, x4)
    x6 = totuple(x5)
    x7 = apply(color, x6)
    x8 = leastcommon(x7)
    x9 = ofcolor(I, x8)
    x10 = inbox(x9)
    x11 = recolor(ZERO, x10)
    x12 = occurrences(I, x11)
    x13 = normalize(x9)
    x14 = shift(x13, NEG_UNITY)
    x15 = lbind(shift, x14)
    x16 = mapply(x15, x12)
    x17 = fill(I, x8, x16)
    return x17
import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_9f236235(I: Grid) -> Grid:
    x0 = frontiers(I)
    x1 = merge(x0)
    x2 = color(x1)
    x3 = shape(I)
    x4 = canvas(x2, x3)
    x5 = hconcat(I, x4)
    x6 = objects(x5, T, F, T)
    x7 = apply(uppermost, x6)
    x8 = order(x7, identity)
    x9 = lbind(sfilter, x6)
    x10 = lbind(matcher, uppermost)
    x11 = compose(x9, x10)
    x12 = lbind(apply, color)
    x13 = rbind(order, leftmost)
    x14 = chain(x12, x13, x11)
    x15 = apply(x14, x8)
    x16 = vmirror(x15)
    return x16
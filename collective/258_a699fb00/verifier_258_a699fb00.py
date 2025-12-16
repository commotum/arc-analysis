import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_a699fb00(I: Grid) -> Grid:
    x0 = leastcolor(I)
    x1 = height(I)
    x2 = vsplit(I, x1)
    x3 = lbind(recolor, TWO)
    x4 = rbind(ofcolor, x0)
    x5 = chain(x3, delta, x4)
    x6 = fork(paint, identity, x5)
    x7 = apply(x6, x2)
    x8 = merge(x7)
    return x8
import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_a3325580(I: Grid) -> Grid:
    x0 = objects(I, T, F, T)
    x1 = valmax(x0, size)
    x2 = sizefilter(x0, x1)
    x3 = order(x2, leftmost)
    x4 = apply(color, x3)
    x5 = astuple(ONE, x1)
    x6 = rbind(canvas, x5)
    x7 = apply(x6, x4)
    x8 = merge(x7)
    x9 = dmirror(x8)
    return x9
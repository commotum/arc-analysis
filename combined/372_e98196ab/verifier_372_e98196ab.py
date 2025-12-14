import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_e98196ab(I: Grid) -> Grid:
    x0 = frontiers(I)
    x1 = sfilter(x0, hline)
    x2 = size(x1)
    x3 = positive(x2)
    x4 = branch(x3, vsplit, hsplit)
    x5 = x4(I, TWO)
    x6 = first(x5)
    x7 = last(x5)
    x8 = fgpartition(x7)
    x9 = merge(x8)
    x10 = paint(x6, x9)
    return x10
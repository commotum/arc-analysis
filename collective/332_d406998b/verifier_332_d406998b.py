import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_d406998b(I: Grid) -> Grid:
    x0 = vmirror(I)
    x1 = fgpartition(x0)
    x2 = merge(x1)
    x3 = toindices(x2)
    x4 = compose(double, halve)
    x5 = fork(equality, identity, x4)
    x6 = compose(x5, last)
    x7 = sfilter(x3, x6)
    x8 = fill(x0, THREE, x7)
    x9 = vmirror(x8)
    return x9
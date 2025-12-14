import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_746b3537(I: Grid) -> Grid:
    x0 = first(I)
    x1 = dedupe(x0)
    x2 = size(x1)
    x3 = equality(ONE, x2)
    x4 = branch(x3, dmirror, identity)
    x5 = x4(I)
    x6 = objects(x5, T, F, F)
    x7 = order(x6, leftmost)
    x8 = apply(color, x7)
    x9 = repeat(x8, ONE)
    x10 = x4(x9)
    return x10
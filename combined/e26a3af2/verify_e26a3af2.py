import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_e26a3af2(I: Grid) -> Grid:
    x0 = rot90(I)
    x1 = apply(mostcommon, I)
    x2 = apply(mostcommon, x0)
    x3 = repeat(x1, ONE)
    x4 = repeat(x2, ONE)
    x5 = compose(size, dedupe)
    x6 = x5(x1)
    x7 = x5(x2)
    x8 = greater(x7, x6)
    x9 = branch(x8, height, width)
    x10 = x9(I)
    x11 = rot90(x3)
    x12 = branch(x8, x4, x11)
    x13 = branch(x8, vupscale, hupscale)
    x14 = x13(x12, x10)
    return x14
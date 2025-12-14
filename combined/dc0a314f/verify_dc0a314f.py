import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_dc0a314f(I: Grid) -> Grid:
    x0 = replace(I, THREE, NEG_ONE)
    x1 = dmirror(x0)
    x2 = papply(pair, x0, x1)
    x3 = lbind(apply, maximum)
    x4 = apply(x3, x2)
    x5 = cmirror(x4)
    x6 = papply(pair, x4, x5)
    x7 = apply(x3, x6)
    x8 = hmirror(x7)
    x9 = papply(pair, x7, x8)
    x10 = apply(x3, x9)
    x11 = vmirror(x10)
    x12 = papply(pair, x11, x10)
    x13 = apply(x3, x12)
    x14 = ofcolor(I, THREE)
    x15 = subgrid(x14, x13)
    return x15
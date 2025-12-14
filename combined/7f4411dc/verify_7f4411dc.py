import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_7f4411dc(I: Grid) -> Grid:
    x0 = objects(I, T, F, F)
    x1 = totuple(x0)
    x2 = apply(color, x1)
    x3 = mostcommon(x2)
    x4 = canvas(x3, TWO_BY_TWO)
    x5 = asobject(x4)
    x6 = palette(I)
    x7 = matcher(identity, x3)
    x8 = argmin(x6, x7)
    x9 = shape(I)
    x10 = canvas(x8, x9)
    x11 = lbind(shift, x5)
    x12 = occurrences(I, x5)
    x13 = mapply(x11, x12)
    x14 = paint(x10, x13)
    return x14
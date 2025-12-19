import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_8f2ea7aa(I: Grid) -> Grid:
    x0 = fgpartition(I)
    x1 = merge(x0)
    x2 = normalize(x1)
    x3 = mostcolor(I)
    x4 = shape(x2)
    x5 = multiply(x4, x4)
    x6 = canvas(x3, x5)
    x7 = shape(x2)
    x8 = rbind(multiply, x7)
    x9 = toindices(x2)
    x10 = apply(x8, x9)
    x11 = lbind(shift, x2)
    x12 = mapply(x11, x10)
    x13 = paint(x6, x12)
    return x13
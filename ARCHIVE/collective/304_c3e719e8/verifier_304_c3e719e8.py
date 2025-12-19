import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_c3e719e8(I: Grid) -> Grid:
    x0 = shape(I)
    x1 = multiply(x0, x0)
    x2 = canvas(ZERO, x1)
    x3 = mostcolor(I)
    x4 = ofcolor(I, x3)
    x5 = lbind(multiply, x0)
    x6 = apply(x5, x4)
    x7 = asobject(I)
    x8 = lbind(shift, x7)
    x9 = mapply(x8, x6)
    x10 = paint(x2, x9)
    return x10
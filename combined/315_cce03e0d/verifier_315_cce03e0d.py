import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_cce03e0d(I: Grid) -> Grid:
    x0 = ofcolor(I, TWO)
    x1 = shape(I)
    x2 = multiply(x1, x1)
    x3 = canvas(ZERO, x2)
    x4 = rbind(multiply, x1)
    x5 = apply(x4, x0)
    x6 = asobject(I)
    x7 = lbind(shift, x6)
    x8 = mapply(x7, x5)
    x9 = paint(x3, x8)
    return x9
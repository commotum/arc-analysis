import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_feca6190(I: Grid) -> Grid:
    x0 = asobject(I)
    x1 = matcher(first, ZERO)
    x2 = compose(flip, x1)
    x3 = sfilter(x0, x2)
    x4 = size(x3)
    x5 = width(I)
    x6 = multiply(x5, x4)
    x7 = multiply(UNITY, x6)
    x8 = canvas(ZERO, x7)
    x9 = multiply(x5, x4)
    x10 = decrement(x9)
    x11 = lbind(astuple, x10)
    x12 = rbind(shoot, UP_RIGHT)
    x13 = compose(last, last)
    x14 = chain(x12, x11, x13)
    x15 = fork(recolor, first, x14)
    x16 = mapply(x15, x3)
    x17 = paint(x8, x16)
    return x17
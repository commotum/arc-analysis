import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_8eb1be9a(I: Grid) -> Grid:
    x0 = fgpartition(I)
    x1 = merge(x0)
    x2 = height(x1)
    x3 = height(I)
    x4 = interval(ZERO, x3, x2)
    x5 = lbind(shift, x1)
    x6 = compose(x5, toivec)
    x7 = compose(x6, invert)
    x8 = fork(combine, x6, x7)
    x9 = mapply(x8, x4)
    x10 = paint(I, x9)
    return x10
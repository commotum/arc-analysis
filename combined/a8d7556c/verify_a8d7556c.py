import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_a8d7556c(I: Grid) -> Grid:
    x0 = initset(ORIGIN)
    x1 = recolor(ZERO, x0)
    x2 = upscale(x1, TWO)
    x3 = occurrences(I, x2)
    x4 = lbind(shift, x2)
    x5 = mapply(x4, x3)
    x6 = fill(I, TWO, x5)
    return x6
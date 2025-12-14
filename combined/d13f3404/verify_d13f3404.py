import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_d13f3404(I: Grid) -> Grid:
    x0 = asobject(I)
    x1 = mostcolor(I)
    x2 = matcher(first, x1)
    x3 = compose(flip, x2)
    x4 = sfilter(x0, x3)
    x5 = apply(initset, x4)
    x6 = rbind(shoot, UNITY)
    x7 = compose(x6, center)
    x8 = fork(recolor, color, x7)
    x9 = mapply(x8, x5)
    x10 = shape(I)
    x11 = double(x10)
    x12 = mostcolor(I)
    x13 = canvas(x12, x11)
    x14 = paint(x13, x9)
    return x14
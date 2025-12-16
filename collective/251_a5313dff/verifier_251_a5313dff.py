import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_a5313dff(I: Grid) -> Grid:
    x0 = objects(I, T, F, F)
    x1 = rbind(bordering, I)
    x2 = compose(flip, x1)
    x3 = sfilter(x0, x2)
    x4 = totuple(x3)
    x5 = apply(color, x4)
    x6 = mostcommon(x5)
    x7 = mostcolor(I)
    x8 = colorfilter(x0, x7)
    x9 = rbind(bordering, I)
    x10 = compose(flip, x9)
    x11 = mfilter(x8, x10)
    x12 = fill(I, ONE, x11)
    return x12
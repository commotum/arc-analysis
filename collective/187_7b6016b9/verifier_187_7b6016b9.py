import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_7b6016b9(I: Grid) -> Grid:
    x0 = objects(I, T, F, F)
    x1 = asindices(I)
    x2 = box(x1)
    x3 = toobject(x2, I)
    x4 = mostcolor(x3)
    x5 = colorfilter(x0, x4)
    x6 = rbind(bordering, I)
    x7 = compose(flip, x6)
    x8 = mfilter(x5, x7)
    x9 = fill(I, TWO, x8)
    x10 = replace(x9, x4, THREE)
    return x10
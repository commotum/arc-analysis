import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_253bf280(I: Grid) -> Grid:
    x0 = leastcolor(I)
    x1 = ofcolor(I, x0)
    x2 = prapply(connect, x1, x1)
    x3 = rbind(greater, ONE)
    x4 = compose(x3, size)
    x5 = sfilter(x2, x4)
    x6 = fork(either, vline, hline)
    x7 = mfilter(x5, x6)
    x8 = fill(I, THREE, x7)
    x9 = leastcolor(I)
    x10 = fill(x8, x9, x1)
    return x10
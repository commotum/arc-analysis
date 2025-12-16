import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_90c28cc7(I: Grid) -> Grid:
    x0 = matcher(identity, ZERO)
    x1 = compose(flip, x0)
    x2 = rbind(sfilter, x1)
    x3 = chain(positive, size, x2)
    x4 = rbind(sfilter, x3)
    x5 = compose(dmirror, x4)
    x6 = power(x5, FOUR)
    x7 = x6(I)
    x8 = dedupe(x7)
    x9 = dmirror(x8)
    x10 = dedupe(x9)
    x11 = dmirror(x10)
    return x11
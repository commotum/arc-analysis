import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_25d8a9c8(I: Grid) -> Grid:
    x0 = width(I)
    x1 = rbind(branch, ZERO)
    x2 = rbind(x1, FIVE)
    x3 = compose(size, dedupe)
    x4 = matcher(x3, ONE)
    x5 = compose(x2, x4)
    x6 = rbind(repeat, x0)
    x7 = compose(x6, x5)
    x8 = apply(x7, I)
    return x8
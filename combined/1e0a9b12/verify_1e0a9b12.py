import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_1e0a9b12(I: Grid) -> Grid:
    x0 = mostcolor(I)
    x1 = rot270(I)
    x2 = matcher(identity, x0)
    x3 = rbind(sfilter, x2)
    x4 = compose(flip, x2)
    x5 = rbind(sfilter, x4)
    x6 = fork(combine, x3, x5)
    x7 = apply(x6, x1)
    x8 = rot90(x7)
    return x8
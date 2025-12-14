import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_bb43febb(I: Grid) -> Grid:
    x0 = objects(I, T, F, F)
    x1 = fork(equality, toindices, backdrop)
    x2 = rbind(greater, ONE)
    x3 = chain(x2, minimum, shape)
    x4 = fork(both, x1, x3)
    x5 = sfilter(x0, x4)
    x6 = compose(backdrop, inbox)
    x7 = mapply(x6, x5)
    x8 = fill(I, TWO, x7)
    return x8
import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_810b9b61(I: Grid) -> Grid:
    x0 = objects(I, T, F, T)
    x1 = rbind(greater, TWO)
    x2 = chain(x1, minimum, shape)
    x3 = fork(equality, toindices, box)
    x4 = fork(both, x2, x3)
    x5 = mfilter(x0, x4)
    x6 = fill(I, THREE, x5)
    return x6
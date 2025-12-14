import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_7447852a(I: Grid) -> Grid:
    x0 = index(I, ORIGIN)
    x1 = shape(I)
    x2 = canvas(x0, x1)
    x3 = hconcat(I, x2)
    x4 = objects(x3, F, F, T)
    x5 = compose(last, centerofmass)
    x6 = order(x4, x5)
    x7 = size(x6)
    x8 = interval(ZERO, x7, ONE)
    x9 = pair(x6, x8)
    x10 = rbind(multiply, THREE)
    x11 = rbind(divide, THREE)
    x12 = chain(x10, x11, last)
    x13 = fork(equality, last, x12)
    x14 = sfilter(x9, x13)
    x15 = mapply(first, x14)
    x16 = fill(I, FOUR, x15)
    return x16
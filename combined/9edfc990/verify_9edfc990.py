import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_9edfc990(I: Grid) -> Grid:
    x0 = objects(I, T, F, F)
    x1 = colorfilter(x0, ZERO)
    x2 = ofcolor(I, ONE)
    x3 = rbind(adjacent, x2)
    x4 = mfilter(x1, x3)
    x5 = recolor(ONE, x4)
    x6 = paint(I, x5)
    return x6
import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_ba97ae07(I: Grid) -> Grid:
    x0 = objects(I, T, F, T)
    x1 = totuple(x0)
    x2 = apply(color, x1)
    x3 = mostcommon(x2)
    x4 = ofcolor(I, x3)
    x5 = backdrop(x4)
    x6 = fill(I, x3, x5)
    return x6
import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_9ecd008a(I: Grid) -> Grid:
    x0 = ofcolor(I, ZERO)
    x1 = rbind(colorcount, ZERO)
    x2 = lbind(toobject, x0)
    x3 = compose(x1, x2)
    x4 = vmirror(I)
    x5 = hmirror(I)
    x6 = astuple(x4, x5)
    x7 = argmin(x6, x3)
    x8 = subgrid(x0, x7)
    return x8
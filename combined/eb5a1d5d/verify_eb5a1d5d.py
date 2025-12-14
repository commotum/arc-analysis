import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_eb5a1d5d(I: Grid) -> Grid:
    x0 = compose(dmirror, dedupe)
    x1 = x0(I)
    x2 = x0(x1)
    x3 = fork(remove, last, identity)
    x4 = compose(hmirror, x3)
    x5 = fork(vconcat, identity, x4)
    x6 = x5(x2)
    x7 = dmirror(x6)
    x8 = x5(x7)
    return x8
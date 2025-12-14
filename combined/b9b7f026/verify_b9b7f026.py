import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_b9b7f026(I: Grid) -> Grid:
    x0 = fgpartition(I)
    x1 = fork(equality, toindices, backdrop)
    x2 = compose(flip, x1)
    x3 = extract(x0, x2)
    x4 = color(x3)
    x5 = canvas(x4, UNITY)
    return x5
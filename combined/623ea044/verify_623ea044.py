import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_623ea044(I: Grid) -> Grid:
    x0 = leastcolor(I)
    x1 = ofcolor(I, x0)
    x2 = rbind(shoot, UNITY)
    x3 = rbind(shoot, NEG_UNITY)
    x4 = fork(combine, x2, x3)
    x5 = rbind(shoot, UP_RIGHT)
    x6 = rbind(shoot, DOWN_LEFT)
    x7 = fork(combine, x5, x6)
    x8 = fork(combine, x4, x7)
    x9 = mapply(x8, x1)
    x10 = fill(I, x0, x9)
    return x10
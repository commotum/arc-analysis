import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_7b7f7511(I: Grid) -> Grid:
    x0 = lefthalf(I)
    x1 = righthalf(I)
    x2 = equality(x0, x1)
    x3 = branch(x2, lefthalf, tophalf)
    x4 = x3(I)
    return x4
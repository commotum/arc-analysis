import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_62c24649(I: Grid) -> Grid:
    x0 = vmirror(I)
    x1 = hconcat(I, x0)
    x2 = hmirror(x1)
    x3 = vconcat(x1, x2)
    return x3
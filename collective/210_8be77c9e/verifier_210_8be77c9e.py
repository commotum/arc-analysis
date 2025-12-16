import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_8be77c9e(I: Grid) -> Grid:
    x0 = hmirror(I)
    x1 = vconcat(I, x0)
    return x1
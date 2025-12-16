import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_4c4377d9(I: Grid) -> Grid:
    x0 = hmirror(I)
    x1 = vconcat(x0, I)
    return x1
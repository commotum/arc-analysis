import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl/utils resolve when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *
from utils import *

def generate_2dee498d(diff_lb: float, diff_ub: float) -> dict:
    dim_bounds = (1, 30)
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 30))
    w = unifint(diff_lb, diff_ub, (1, 10))
    bgc = choice(cols)
    go = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (1, min(9, h * w)))
    colsch = sample(remcols, numc)
    inds = totuple(asindices(go))
    for col in colsch:
        num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = sample(inds, num)
        go = fill(go, col, chos)
        inds = difference(inds, chos)
    gi = hconcat(go, hconcat(go, go))
    return {'input': gi, 'output': go}
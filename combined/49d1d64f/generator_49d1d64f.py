import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl/utils resolve when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *
from utils import *

def generate_49d1d64f(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 28))
    w = unifint(diff_lb, diff_ub, (2, 28))
    ncols = unifint(diff_lb, diff_ub, (1, 10))
    ccols = sample(cols, ncols)
    gi = canvas(-1, (h, w))
    obj = {(choice(ccols), ij) for ij in asindices(gi)}
    gi = paint(gi, obj)
    go = canvas(0, (h+2, w+2))
    go = paint(go, shift(asobject(gi), (1, 1)))
    ts = sfilter(obj, lambda cij: cij[1][0] == 0)
    bs = sfilter(obj, lambda cij: cij[1][0] == h - 1)
    ls = sfilter(obj, lambda cij: cij[1][1] == 0)
    rs = sfilter(obj, lambda cij: cij[1][1] == w - 1)
    ts = shift(ts, (1, 1))
    bs = shift(bs, (1, 1))
    ls = shift(ls, (1, 1))
    rs = shift(rs, (1, 1))
    go = paint(go, shift(ts, (-1, 0)))
    go = paint(go, shift(bs, (1, 0)))
    go = paint(go, shift(ls, (0, -1)))
    go = paint(go, shift(rs, (0, 1)))
    return {'input': gi, 'output': go}
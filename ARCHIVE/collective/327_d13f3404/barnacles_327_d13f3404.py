"""
# [327] d13f3404.json
* image_expansion
* draw_line_from_point
* diagonals
"""

def p(g,e=enumerate):X=[[0]*6 for _ in[0]*6];[X[r+i].__setitem__(c+i,v)for r,R in e(g)for c,v in e(R)if v for i in range(6-max(r,c))];return X
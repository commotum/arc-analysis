"""
# [299] bdad9b1f.json
* draw_line_from_point
* direction_guessing
* recoloring
* take_intersection
"""

def p(j):A=len(j)//2;c=[j[i][i]for i in range(A)];E={c[i]:c[i-1]for i in range(A)};return[[E[i]for i in r]for r in j]
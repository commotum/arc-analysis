"""
# [212] 8d510a79.json
* draw_line_from_point
* detect_wall
* direction_guessing
* associate_colors_to_bools
"""

def p(g,L=len,R=range):
 h,w=L(g),L(g[0])
 W=[i for i in R(L(g)) if 5 in g[i]][0]
 for r in R(h):
  for c in R(w):
   if g[r][c]==1 and r<W:
    for z in R(r,-1,-1):g[z][c]=1
   elif g[r][c]==1 and r>W:
    for z in R(r,h):g[z][c]=1
   if g[r][c]==2 and r<W:
    for z in R(r,W):g[z][c]=2
   elif g[r][c]==2 and r>W:
    for z in R(W+1,r):g[z][c]=2
 return g
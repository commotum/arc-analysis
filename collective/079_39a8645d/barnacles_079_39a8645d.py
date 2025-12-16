"""
# [080] 39e1d7f9.json
* detect_grid
* pattern_repetition
* grid_coloring
"""

def p(g):
 #color count trick not working need to count 3x3 shapes with max color
 f=sum(g,[])
 C=sorted([[f.count(c),c] for c in set(f)])
 C=C[-2][1]
 g=[[c if c in [0,C] else 0 for c in r] for r in g]
 for r in range(len(g)):
  if C in g[r]:
   i=g[r].index(C)
   if g[r+1][i-1]==C:i-=1
   g=[y[i:i+3] for y in g[r:r+3]]
   break
 return g
"""
# [308-R] c8cbb738.json
* pattern_moving
* jigsaw
* crop
"""

def p(g):
 X=[]
 for r in g:
  for i in range(2):
   X+=[sum([[c]*2 for c in r],[])]
 return X
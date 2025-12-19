"""
# [223] 9172f3a0.json
* image_resizing
"""

def p(g):
 X=[]
 for r in g:
  for i in range(3):
   X+=[sum([[c]*3 for c in r],[])]
 return X
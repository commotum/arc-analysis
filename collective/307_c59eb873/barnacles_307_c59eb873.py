"""
# [307] c59eb873.json
* image_resizing
"""

def p(g):
 X=[]
 for r in g:
  for i in range(2):
   X+=[sum([[c]*2 for c in r],[])]
 return X
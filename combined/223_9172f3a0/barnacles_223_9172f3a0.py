"""
# [224] 928ad970.json
* rectangle_guessing
* color_guessing
* draw_rectangle
"""

def p(g):
 X=[]
 for r in g:
  for i in range(3):
   X+=[sum([[c]*3 for c in r],[])]
 return X
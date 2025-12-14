"""
# [304] c3e719e8.json
* image_repetition
* image_expansion
* count_different_colors
* take_maximum
"""

def p(j,A=range):
 c,E=len(j),len(j[0])
 for k in A(c):
  if sum(j[k])==0:j[k]=[2]*E
 for W in A(E):
  if all(j[k][W]in[0,2]for k in A(c)):
   for k in A(c):j[k][W]=2
 return j
"""
# [233] 97a05b5b.json
* pattern_moving
* pattern_juxtaposition
* crop
* shape_guessing
"""

def p(j,A=enumerate):
 for c,E in A(j):
  k,W,l=0,[],0
  for J,a in A(E):
   if a>0:W=[a,5]*20;l=1
   if l:j[c][J]=W[k];k+=1
 return j
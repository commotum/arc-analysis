"""
# [386] f2829549.json
* detect_wall
* separate_images
* take_complement
* pattern_intersection
"""

def p(j):
 for A in range(4):
  for c in range(3):
   j[A][c]+=j[A][c+4]
   if j[A][c]>0:j[A][c]=0
   else:j[A][c]=3
 return[R[:3]for R in j]
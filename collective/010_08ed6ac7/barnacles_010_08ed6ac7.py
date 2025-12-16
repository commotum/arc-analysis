"""
# [011] 09629e4f.json
* detect_grid
* separate_images
* count_tiles
* take_minimum
* enlarge_image
* create_grid
* adapt_image_to_grid
"""

def p(j):
 A={}
 for c in j:
  for E,k in enumerate(c):
   if k==5:c[E]=A.setdefault(E,len(A)+1)
 return j
"""
# [275] b190f7f5.json
* separate_images
* image_expasion
* color_palette
* image_resizing
* replace_pattern
"""

def p(j):
 A=min(len(j),len(j[0]));p,c=[r[:A]for r in j[:A]],[r[-A:]for r in j[-A:]]
 if any(max(r)==8 for r in p):p,c=c,p
 return[[p[y//A][x//A]*c[y%A][x%A]//8 for x in range(A*A)]for y in range(A*A)]
"""
# [275] b190f7f5.json
* separate_images
* image_expasion
* color_palette
* image_resizing
* replace_pattern
"""

j=lambda A,c:sum(sum(i==c for i in r)for r in A)
def p(A):E=max(j([r],8)for r in A);k=(j(A,5)-E-2)/2-j(A,8)/E;return[[8*(k>0),8*(k>1),8*(k>2)],[0,0,8*(k>3)],[0,0,0]]
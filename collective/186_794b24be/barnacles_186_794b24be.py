"""
# [186] 794b24be.json
* count_tiles
* associate_images_to_numbers
"""

p=lambda j,A=[2]*3,c=[0]*3:[[A,[0,2,0],c],[A,c,c],[[2,2,0],c,c],[[2,0,0],c,c]][4-sum(r.count(1)for r in j)]
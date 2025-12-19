"""
# [332] d406998b.json
* recoloring
* one_yes_one_no
* cylindrical
"""

p=lambda g:[[3if g[i][j]==5and(len(g[0])-1-j)%2==0else g[i][j]for j in range(len(g[0]))]for i in range(3)]
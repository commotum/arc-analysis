"""
# [262] a85d4709.json
* separate_images
* associate_colors_to_images
* summarize
"""

def p(j):j=[j[-1]]+j[:len(j)-1];j=[[2 if C==8 else C for C in R]for R in j];return j
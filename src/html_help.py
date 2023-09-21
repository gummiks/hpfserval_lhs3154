import numpy as np
import pandas as pd

def html_img(filename,height=0,width=0):
    """
    Return html code for an image
    
    EXAMPLE: 
        html_img('/test/a.png',1,2)
    """
    str_height=''
    str_width=''
    if height!=0:
        str_height="height='{}px' ".format(height)
    if width!=0:
        str_width="width='{}px' ".format(width)
    code = """<img {}{}src='{}'>""".format(str_height,str_width,filename)
    print('HTML: {}'.format(code))
    return code

import cv2 as cv
import numpy as np
def edge_filter(img,bShow=True):
    """
    Viet ham de lam trac nghiem
    :param img:
    :param bShow:
    :return:
    """
    kernel1 = np.array([
        [0,-1,-1],
        [1, 0,-1],
        [1, 1, 0]
    ])
    kernel2 = np.array([
        [0,  1, 1],
        [-1, 0, 1],
        [-1,-1, 0]]
    )
    if bShow:
        edge1=cv.filter2D(src=img,ddepth=-1,kernel=kernel1)
        edge2 = cv.filter2D(src=img, ddepth=-1, kernel=kernel2)
        cv.imshow("src",img)
        cv.imshow("dest1",edge1)
        cv.imshow("dest2",edge2)
        cv.waitKey(0)


img = cv.imread("pipe.jpg")
edge_filter(img)
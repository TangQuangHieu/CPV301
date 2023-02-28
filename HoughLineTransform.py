import cv2 as cv
import numpy as np
import math
def hough_line_transform_demo():
    """
    https://aicurious.io/blog/2019-10-24-hough-transform-phat-hien-duong-thang
    https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
    :return:
    """
    img = cv.imread("lane.png")
    img_clone = np.copy(img)
    cv.imshow("src",img)
    img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #Bo loc canh Canny,
    #minVal: Quyet dinh khong phai la canh
    #MaxVal: Quyet dinh la canh
    #Giua min va max, dung ket noi canh de kiem tra
    edges = cv.Canny(img_gray,100,200)
    cv.imshow("Canh",edges)
    cv.waitKey(0)
    lines = cv.HoughLines(edges,1,np.pi/180,110,None,0,0)

    #1: do phan giai cua p theo pixel
    #np.pi/180: Do phan giai cua theta theo do( 1 do)
    #150: Nguong dem de chap nhan cac diem thuoc canh
    #0,0: su dung hough nguyen ban, khong thay doi

    for i in range(0,len(lines)):
        #rho = x*cos(theta)+y*sin(theta)
        #rho = a*x + b*y
        #a*(x0-1000b)+b*(y0+1000*a)=rho
        print(lines[i])
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a*rho
        y0 = b*rho
        pt1 = (int(x0-1000*b),int(y0+1000*a))
        pt2 = (int(x0+1000*b),int(y0-1000*a))
        cv.line(img_clone,pt1,pt2,(0,0,255),1,cv.LINE_AA)
    cv.imshow("lines",img_clone)
    cv.waitKey(0)
    cv.destroyAllWindows()


win_name = "linesP"
img = cv.imread("lane.png")
def hough_lineP_transform_callback(val):
    img_clone = np.copy(img)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(img_gray, 100, 200)
    threshold = cv.getTrackbarPos('threshold',win_name)
    minLinLength = cv.getTrackbarPos('minLinLength', win_name)
    maxLineGap = cv.getTrackbarPos('maxLineGap', win_name)
    linesP = cv.HoughLinesP(edges, 1, np.pi / 180, threshold, None, minLinLength, maxLineGap)
    # 1: do phan giai cua p theo pixel
    # np.pi/180: Do phan giai cua theta theo do( 1 do)
    # 150: Nguong dem de chap nhan cac diem thuoc canh
    # 0,0: su dung hough nguyen ban, khong thay doi

    if linesP is not None:
        for i in range(len(linesP)):
            l = linesP[i][0]
            cv.line(img_clone, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 1, cv.LINE_AA)
    cv.imshow("linesP", img_clone)
    cv.waitKey(1)

def hough_lineP_transform_demo():
    """
    https://aicurious.io/blog/2019-10-24-hough-transform-phat-hien-duong-thang
    https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
    :return:
    """

    cv.imshow('src',img)
    cv.waitKey(1)

    cv.namedWindow(win_name)

    cv.createTrackbar('threshold',win_name,0,250,hough_lineP_transform_callback)
    cv.createTrackbar('minLinLength', win_name, 0, 100, hough_lineP_transform_callback)
    cv.createTrackbar('maxLineGap', win_name, 0, 200,hough_lineP_transform_callback)
    # Show some stuff
    # hough_lineP_transform_callback()
    cv.waitKey(0)
def main():
    hough_line_transform_demo()
    #hough_lineP_transform_demo()
main()
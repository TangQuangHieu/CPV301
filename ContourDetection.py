import cv2 as cv
import numpy as np

def find_contour(img,retrieve_type=cv.RETR_LIST):
    """

    :param retrieve_type:
        * RETR_LIST: Chỉ trả về tập các đường viền không có thứ bậc
        * RETR_EXTERNAL: Chỉ trả về viền ngoài nếu có 2 viền
        * RETR_CCOMP: trả về tất cả các viền, các viền ngoài
        thì có bậc 1, viền trong thì có bậc 2
        * RETR_TREE: trả về tất cả các viền với các mức thứ
        bậc hoàn chỉnh, có thể lớn hơn 2
        Giới hạn: Hàm làm việc không tốt với các
        hình mà màu vật thể gần giống màu nền, xem thêm link
        để tham khảo
        danh sách thứ bậc: [next,prev,child1,parent]
    :return: trả về contours và danh sách thứ bậc
    """
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(img_gray, 150, 255, cv.THRESH_BINARY)
    # Không tạo cây cha con cho các viền, nhanh
    contours, hierarchy = cv.findContours(thresh, retrieve_type, cv.CHAIN_APPROX_NONE)
    #img_draw = np.copy(img)
    #img_draw = cv.drawContours(img_draw, contours, -1, (0, 255, 0), 2, cv.LINE_AA)
    return [contours,hierarchy]

def draw_contours(img,contours,hierarchy,win_name):
    """
    :param img_draw: hình để vẽ viền
    :param contours: các danh sách viền
    :param hierarchy: thứ bậc của viền
    :param win_name: tên cửa sồ hình vẽ
    :return:
    """
    img_draw = np.copy(img)
    #img_draw = cv.drawContours(img_draw, contours, -1, (0, 255, 0), 2, cv.LINE_AA)

    colors=[(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255)]
    #Line 0: màu xanh nước biển
    #Line 1: màu lục
    #line 2: màu đỏ
    #Line 3: Màu cam
    #Line 4: Màu hồng
    for i in range(len(contours)):
        img_draw = cv.drawContours(img_draw, [contours[i]], -1, colors[i], 2, cv.LINE_AA)
    print(f"{win_name}: {hierarchy}")
    cv.imshow(win_name,img_draw)
    cv.waitKey(0)

def contour_detection_demo():
    """
    https://learnopencv.com/contour-detection-using-opencv-python-c/
    :return:
    """
    """
    Contour detection and drawing using different extraction modes to complement
    the understanding of hierarchies
    """
    img = cv.imread('countour.jpg')
    cv.imshow("src",img)
    #img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    contours,hierarchy = find_contour(img,cv.RETR_LIST)
    draw_contours(img,contours,hierarchy,"LIST METHOD")

    contours, hierarchy = find_contour(img, cv.RETR_EXTERNAL)
    draw_contours(img, contours, hierarchy, "EXTERNAL METHOD")

    contours, hierarchy = find_contour(img, cv.RETR_CCOMP)
    draw_contours(img, contours, hierarchy, "CCOMP METHOD")

    contours, hierarchy = find_contour(img, cv.RETR_TREE)
    draw_contours(img, contours, hierarchy, "TREE METHOD")

def main():
    contour_detection_demo()

main()









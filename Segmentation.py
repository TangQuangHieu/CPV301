import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def waterShedDemo():
    """
    Demo for water shed algorithm
    Refereces:
    https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html
    https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    erosion: phép ăn mòn
    dilate: phép dãn
    MORPH_OPEN: erosion sau đó dilate: loại bỏ nhiễu trắng
    CLOSE: dilate sau đó erosion: Loại bỏ lỗ backgound trong foreground
    :return:
    """

    img = cv.imread('water_coins.jpg')
    cv.imshow("src",img)
    cv.waitKey(0)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    cv.imshow("binary image",thresh)
    cv.waitKey(0)

    # Loại bỏ nhiễu
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
    cv.imshow("OPEN",opening)
    cv.waitKey(0)

    # Mở rộng hình, đảm bảo phần màu đen chắc chắn là background
    sure_bg = cv.dilate(opening, kernel, iterations=3)
    cv.imshow("sure_bg",sure_bg)

    # Biến đổi khoảng cách( tham khảo), sau đó
    # Lấy theshold để đảm bảo là object không bị dính liền
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    cv.imshow("dist",dist_transform)
    cv.waitKey(0)
    ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    cv.imshow("sure_fg",sure_fg)
    cv.waitKey(0)

    # Tìm vùng không thể phân biệt
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)
    cv.imshow("unk",unknown)
    cv.waitKey(0)

    # Đánh dấu vùng chắc chắn là foreground ( objects)
    ret, markers = cv.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, Đánh dấu vùng không biết là 0
    markers[unknown == 255] = 0
    print(markers.shape)


    #Vẽ marker dùng matplotlib
    fig, ax = plt.subplots()
    ax.imshow(markers,cmap="PuOr")
    ax.set_title("marker")
    plt.tight_layout()
    plt.show()

    #Áp dụng thuật toán watershed cho hình gốc
    # Đường viền của object sẽ được đánh dấu là -1.
    markers = cv.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]
    cv.imshow("dest",img)
    cv.waitKey(0)

def kMeanClusteringDemo():
    """
    use kmean to cluster color in a image
    color quantization
    https://www.geeksforgeeks.org/image-segmentation-using-k-means-clustering/
    :return:
    """
    img=cv.imread("butterfly.png")
    cv.imshow("src",img)
    cv.waitKey(0)
    pixel_vals=img.reshape((-1,3))
    pixel_vals=np.float32(pixel_vals)

    # the below line of code defines the criteria for the algorithm to stop running,
    # which will happen is 100 iterations are run or the epsilon (which is the required accuracy)
    # becomes 85%
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.85)

    # then perform k-means clustering with number of clusters defined as 3
    # also random centres are initially choosed for k-means clustering
    k = 3
    retval, labels, centers = cv.kmeans(pixel_vals, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    # convert data into 8-bit values
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]

    # reshape data into the original image dimensions
    segmented_image = segmented_data.reshape((img.shape))

    cv.imshow("dst",segmented_image)
    cv.waitKey(0)

def meanShiftDemo():
    """
    https://theailearner.com/2019/02/12/2d-histogram/
    https://theailearner.com/tag/cv2-calcbackproject/
    :return:
    """
    cap = cv.VideoCapture("slow_traffic_small.mp4")
    # take first frame of the video
    ret, frame = cap.read()
    # setup initial location of window
    x, y, w, h = 300, 200, 100, 50  # simply hardcoded the values
    track_window = (x, y, w, h)
    # set up the ROI for tracking
    roi = frame[y:y + h, x:x + w]
    hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    #print(hsv_roi[:,:,0].shape)
    #cv.imshow("hsv",hsv_roi[:,:,0])
    #cv.waitKey(0)
    mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    print(roi_hist)
    cv.waitKey(0)
    cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
    # Setup the termination criteria, either 10 iteration or move by at least 1 pt
    term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
    while (1):
        ret, frame = cap.read()
        if ret == True:
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            cv.imshow("match Hist",dst)
            cv.waitKey(30)
            # apply meanshift to get the new location
            ret, track_window = cv.meanShift(dst, track_window, term_crit)
            # Draw it on image
            x, y, w, h = track_window
            img2 = cv.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
            cv.imshow('img2', img2)
            k = cv.waitKey(30) & 0xff
            if k == 27:
                break
        else:
            break

def camShiftDemo():
    """
    https://docs.opencv.org/3.4/d7/d00/tutorial_meanshift.html
    :return:
    """
    cap = cv.VideoCapture("slow_traffic_small.mp4")
    # take first frame of the video
    ret, frame = cap.read()
    # setup initial location of window
    x, y, w, h = 300, 200, 100, 50  # simply hardcoded the values
    track_window = (x, y, w, h)
    # set up the ROI for tracking
    roi = frame[y:y + h, x:x + w]
    hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
    # Setup the termination criteria, either 10 iteration or move by at least 1 pt
    term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
    while (1):
        ret, frame = cap.read()
        if ret == True:
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            # apply camshift to get the new location
            ret, track_window = cv.CamShift(dst, track_window, term_crit)
            # Draw it on image
            pts = cv.boxPoints(ret)
            pts = np.int0(pts)
            img2 = cv.polylines(frame, [pts], True, 255, 2)
            cv.imshow('img2', img2)
            k = cv.waitKey(30) & 0xff
            if k == 27:
                break
        else:
            break

def main():
    #waterShedDemo()
    #kMeanClusteringDemo()
    #meanShiftDemo()
    #camShiftDemo()

main()


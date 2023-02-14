import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def sift_detector_demo():
    """
    https://docs.opencv.org/3.4/da/df5/tutorial_py_sift_intro.html
    :return:
    """
    img = cv.imread('home.jpg')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp = sift.detect(gray, None)
    img = cv.drawKeypoints(img, kp, img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv.imshow('sift_keypoints', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def harris_detector_demo():
    """
    https://docs.opencv.org/3.4/dc/d0d/tutorial_py_features_harris.html
    OpenCV has the function cv.cornerHarris(img,blockSize,ksize,k) for this purpose. Its arguments are:
        - img - Input image. It should be grayscale and float32 type.
        - blockSize - It is the size of neighbourhood considered for corner detection
        - ksize - Aperture parameter of the Sobel derivative used.
        - k - Harris detector free parameter in the equation.
    :return:
    """
    filename = 'sudoku.jpg'
    img = cv.imread(filename)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray, 2, 3, 0.04)
    # result is dilated for marking the corners, not important
    #cv.imshow('feature map', dst)
    #cv.waitKey(0)
    dst = cv.dilate(dst, None)
    #cv.imshow('feature map dilate', dst)
    #cv.waitKey(0)
    # Threshold for an optimal value, it may vary depending on the image.
    img[dst > 0.01 * dst.max()] = [0, 0, 255]
    cv.imshow('dst', img)
    cv.waitKey(0)
    #cv.destroyAllWindows()

def harris_subpixel_demo():
    """
    https://docs.opencv.org/3.4/dc/d0d/tutorial_py_features_harris.html
    To refine the result of harris_detector, we use this function

    void cv::cornerSubPix	(	InputArray 	image,
    InputOutputArray 	corners,
    Size 	winSize,
    Size 	zeroZone,
    TermCriteria 	criteria
    )
    Python:
    cv.cornerSubPix(	image, corners, winSize, zeroZone, criteria	) ->	corners
    :return:
    """
    #filename = 'chessboard2.jpg'
    img = cv.imread("sudoku.jpg")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # find Harris corners
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray, 2, 3, 0.04)
    dst = cv.dilate(dst, None)
    ret, dst = cv.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)
    # find centroids
    ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)
    # define the criteria to stop and refine the corners
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
    # Now draw them
    res = np.hstack((centroids, corners))
    res = np.int0(res)
    img[res[:, 1], res[:, 0]] = [0, 0, 255]
    img[res[:, 3], res[:, 2]] = [0, 255, 0]
    #cv.imwrite('subpixel5.png', img)
    cv.imshow("refined corner",img)
    cv.waitKey(0)

def hog_detector_demo_opencv():
    """
    https://phamdinhkhanh.github.io/2019/11/22/HOG.html
    https://phamdinhkhanh.github.io/2019/11/22/HOG.html
    :return:
    """
    img = cv.imread("messi5.jpg")
    # 1. Khai bao tham so
    cell_size=(8,8)  # kich thuoc 1 cell tinh theo pixel
    block_size=(2,2) # tinh theo cell
    nbins = 9 # So luong bin cua gradient (tu 0 den 180 do)
    # 2. Tinh toan cac tham so truyen vao HOG
    # winSize: Kich thuoc buc anh duoc corp de chia het cho cell size
    winSize = (img.shape[1]//cell_size[1]*cell_size[1],img.shape[0]//cell_size[0]*cell_size[0])
    #BlockSize: Kich thuoc cua 1 block tinh theo pixel
    blockSize = (block_size[1]*cell_size[1],block_size[0]*cell_size[0]) #tinh theo pixel
    blockStride = (cell_size[1],cell_size[0]) #Buoc nhay tinh theo pixel
    print('Kích thước bức ảnh crop theo winSize (pixel): ', winSize)
    print('Kích thước của 1 block (pixel): ', blockSize)
    print('Kích thước của block stride (pixel): ', blockStride)

    # Kích thước của lưới ô vuông.
    n_cells = (img.shape[0] // cell_size[0], img.shape[1] // cell_size[1])
    # 3.Tinh toan HOG descriptor
    hog = cv.HOGDescriptor(_winSize=winSize,
                            _blockSize=blockSize,
                            _blockStride=blockStride,
                            _cellSize=cell_size,
                            _nbins=nbins)

    # Reshape hog feature
    hog_feats = hog.compute(img) \
        .reshape(n_cells[1] - block_size[1] + 1,
                 n_cells[0] - block_size[0] + 1,
                 block_size[0], block_size[1], nbins) \
        .transpose((1, 0, 2, 3, 4))

    print('Kích thước hog feature (h, w, block_size_h, block_size_w, nbins): ', hog_feats.shape)
    hog_visualization(img)
def hog_detector_demo_sklearn():
    """

    :return:
    """
    from skimage import feature
    img = cv.imread("messi5.jpg")
    if len(img.shape) !=2:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    H = feature.hog(img, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2")

    print('Kích thước hog features: ', H.shape)
    hog_visualization(img)
def hog_visualization(img):
    from skimage import exposure
    from skimage import feature

    import matplotlib.pyplot as plt
    #if len(img.shape) != 2:
    #    img = cv.cvtcolor(img, cv.COLOR_BGR2GRAY)
    (H, hogImage) = feature.hog(img, orientations=9, pixels_per_cell=(8, 8),
                                cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2",
                                visualize=True)

    hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
    hogImage = hogImage.astype("uint8")
    plt.imshow(hogImage)
    plt.show()

def hog_human_detection():
    """
    https://phamdinhkhanh.github.io/2019/11/22/HOG.html
    :return:
    """
    from imutils.object_detection import non_max_suppression
    from imutils import paths
    # Khởi tạo một bộ mô tả đặc trưng HOG
    hog = cv.HOGDescriptor()
    hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
    vid = cv.VideoCapture(0)
    fps = 0.
    import time
    # font
    font = cv.FONT_HERSHEY_SIMPLEX
    # fontScale
    fontScale = 1
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2
    while(True):
        # Capture the video frame
        # by frame
        start_time = time.time()
        ret, frame = vid.read()
        (rects, weights) = hog.detectMultiScale(img=frame, winStride=(4, 4),
                                                padding=(8, 8), scale=1.05)
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        # Sử dụng non max suppression để lấy ra bounding box cuối cùng với ngưỡng threshold = 0.65
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)


        for (x, y, w, h) in pick:
            frame = cv.rectangle(frame,(x,y),(x+w,y+h),color,thickness)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        fps = 1. / (time.time() - start_time) #Frame per sec

        frame = cv.putText(frame, str(fps), (20, 20),
                           font, fontScale, color, thickness, cv.LINE_AA)
        cv.imshow("video", frame)
        cv.waitKey(10)
    vid.release()
    cv.destroyAllWindows()

def hog_human_face_detection_dlib():
    """
    https://www.pandaml.com/computer%20vision/practical/nhan-dien-khuon-mat-hog-deeplearning/
    :return:
    """
    import dlib
    def hog_face_to_points(rect):
        x1 = rect.left()
        y1 = rect.top()
        x2 = rect.right()
        y2 = rect.bottom()

        return x1, y1, x2, y2
    vid = cv.VideoCapture(0)
    fps = 0.
    import time
    # font
    font = cv.FONT_HERSHEY_SIMPLEX
    # fontScale
    fontScale = 1
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2
    while (True):

        start_time = time.time()
        ret, frame = vid.read()
        # chuyển sang ảnh rgb (yêu cầu đầu vào của mô hình dlib)
        rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # load mô hình nhận diện khuôn mặt
        hog_face_detector = dlib.get_frontal_face_detector()
        # nhận diện khuôn mặt trong ảnh
        faces = hog_face_detector(rgb_image, upsample_num_times=0)
        # vẽ đường bao cho từng khuôn mặt
        green_color = (0, 255, 0)
        for face in faces:
            x1, y1, x2, y2 = hog_face_to_points(face)
            cv.rectangle(frame, pt1=(x1, y1), pt2=(x2, y2), color=green_color, thickness=2)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        fps = 1. / (time.time() - start_time)  # Frame per sec
        frame = cv.putText(frame, str(fps), (20, 20),
                           font, fontScale, color, thickness, cv.LINE_AA)
        cv.imshow("video", frame)
        cv.waitKey(10)
def main():
    #sift_detector_demo()
    #harris_detector_demo()
    #harris_subpixel_demo()
    #cv.destroyAllWindows()
    #hog_detector_demo_sklearn()
    #hog_human_detection()
    #hog_human_face_detection_dlib()
main()


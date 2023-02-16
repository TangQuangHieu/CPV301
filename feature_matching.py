import cv2 as cv
import numpy as np

def create_rotated_img(img,rot_angle,scale):
    """

    :param img: original image
    :param rot_angle: rotate angle ( in degree)
    :param scale: tuple to scale
    :return: image scaled and rotated
    """
    new_size = (int(scale[0] * img.shape[0]), int(scale[1]*img.shape[1]))
    print(new_size)
    img2 = cv.resize(img, new_size)
    rows, cols, _ = img2.shape
    center = ((cols - 1) / 2.0, (rows - 1) / 2.0)
    M = cv.getRotationMatrix2D(center, rot_angle, 1)
    img2 = cv.warpAffine(img2, M, (cols, rows))
    return img2

def sift_matching_demo():
    """
    Link tham khảo theo thứ tự:
        https://www.analyticsvidhya.com/blog/2019/10/detailed-guide-powerful-sift-technique-image-matching-python/
        https://www.analyticsvidhya.com/blog/2019/10/detailed-guide-powerful-sift-technique-image-matching-python/
        https://medium.com/analytics-vidhya/opencv-feature-matching-sift-algorithm-scale-invariant-feature-transform-16672eafb253
    :return:
    """

    img1 = cv.imread("home.jpg")
    img2 = create_rotated_img(img1, 60, (0.5, 0.5))
    cv.imshow("src",img1)
    cv.imshow("dst",img2)
    cv.waitKey(0)
    cv.destroyAllWindows()

    #Tạo sift object
    #sift = cv.xfeatures2d.SITF_create(): Dùng cho opencv cũ
    sift = cv.SIFT_create()

    #Tìm keypoint cho ảnh 1
    img1_gray = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)

    kp1,descriptors_1 = sift.detectAndCompute(img1_gray,None)
    img1_kp = cv.drawKeypoints(img1_gray, kp1, img1_gray, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imshow('img1_Keypoint', img1_kp)

    #Tìm keypoint cho ảnh 2
    img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    kp2, descriptors_2 = sift.detectAndCompute(img2_gray, None)
    img2_kp = cv.drawKeypoints(img2_gray, kp2, img2_gray, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imshow('img2_Keypoint', img2_kp)
    cv.waitKey(0)
    cv.destroyAllWindows()

    #Matching dùng brute-force
    bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors_1,descriptors_2)
    matches = sorted(matches,key = lambda x:x.distance)
    img_match = cv.drawMatches(img1,kp1,img2,kp2,matches[:20],img2,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imshow("Match",img_match)
    cv.waitKey(0)
    cv.destroyAllWindows()

def orb_matching_demo():
    """
    Link tham khảo theo thứ tự:
        https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
    :return:
    """
    img1 = cv.imread("home.jpg")
    img2 = create_rotated_img(img1, 60, (0.5, 0.5))
    cv.imshow("src",img1)
    cv.imshow("dst",img2)
    cv.waitKey(0)
    cv.destroyAllWindows()

    #Tạo orb object
    #sift = cv.xfeatures2d.SURF_create(): Dùng cho opencv cũ
    orb = cv.ORB_create()

    #Tìm keypoint cho ảnh 1
    img1_gray = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)

    kp1,descriptors_1 = orb.detectAndCompute(img1_gray,None)
    #img1_kp = cv.drawKeypoints(img1, kp1, img1, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #cv.imshow('img1_Keypoint', img1_kp)

    #Tìm keypoint cho ảnh 2
    img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    kp2, descriptors_2 = orb.detectAndCompute(img2_gray, None)
    #img2_kp = cv.drawKeypoints(img2, kp2, img2, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #cv.imshow('img2_Keypoint', img2_kp)
    #cv.waitKey(0)
    #cv.destroyAllWindows()

    #Matching dùng brute-force
    bf = cv.BFMatcher()

    matches = bf.knnMatch(descriptors_1,descriptors_2,k=2)
    #matches = sorted(matches,key = lambda x:x.distance)
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    img_match = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imshow("Match",img_match)
    cv.waitKey(0)
    cv.destroyAllWindows()

def flann_matches():
    """
    https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
    :return:
    """

    img1 = cv.imread("home.jpg")
    img2 = create_rotated_img(img1, 60, (0.5, 0.5))
    img1_gray = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
    img2_gray = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv.DrawMatchesFlags_DEFAULT)
    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
    cv.imshow("Flann Matching",img3)
    cv.waitKey(0)
    cv.destroyAllWindows()


def main():
    #sift_matching_demo()
    #orb_matching_demo()
    flann_matches()

main()





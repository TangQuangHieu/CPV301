import cv2 as cv
import numpy as np
import random

def salt_and_pepper_noise(img,bShow=False):
    """
    add noise salt and pepper noise into image
    :param img: input image
    :param bShow: whether to show the image or not
    :return: noised image
    """
    row, col = img.shape
    num_of_pixel = row*col
    noised_img = np.array(img)
    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(int(num_of_pixel/15), int(num_of_pixel/14))
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to white
        noised_img[y_coord][x_coord] = 255

    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(int(num_of_pixel/15), int(num_of_pixel/14))
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to black
        noised_img[y_coord][x_coord] = 0
    if bShow:
        cv.imshow("src",img)
        cv.imshow("noise img",noised_img)
        cv.waitKey(0)
    return noised_img

def medianFilter(img,size=3,bShow=False):
    """
    Median filter( specilise for salt-pepper noise)
    :param img: input image
    :param size: kernel size
    :return: denoise image
    """
    assert len(img.shape)==2, "Input must be a gray image"
    rows,cols = img.shape
    new_img = np.zeros(img.shape,np.uint8)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            pixel_values = []
            #print(pixel_values,int(-size/2),int(size/2+0.5))
            for i in range(int(-size/2),int(size/2+0.5)):
                for j in range(int(-size/2),int(size/2+0.5)):
                    neighbor_row = row+i
                    neighbor_col = col+j
                    if neighbor_row>=0 and neighbor_col>=0 \
                            and neighbor_row < rows and neighbor_col < cols:
                        pixel_values.append(img[neighbor_row,neighbor_col])
            pixel_values = np.sort(np.array(pixel_values))
            med_idx = len(pixel_values)//2
            new_img[row,col] = pixel_values[med_idx]
    if bShow:
        cv.imshow("src", img)
        cv.imshow("median filter img", new_img)
        cv.waitKey(0)
    return new_img


def medianFilter_demo():
    """
    Demo for median filtering for salt-pepper noised image
    :return:
    """
    img = cv.imread("messi5.jpg", cv.IMREAD_GRAYSCALE)
    noised_img = salt_and_pepper_noise(img)
    medianFilter(noised_img, size=3, bShow=True)


def meanFilter(img,size=3,bShow=False):
    """
    Filtering image using average pixel values

    :param img: noise image
    :param size: size of kernel
    :param bShow: whether to show image or not
    :return: denoise image
    """
    assert len(img.shape)==2, "Input must be a gray image"
    rows,cols = img.shape
    new_img = np.zeros(img.shape,np.uint8)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            sum = 0
            #print(pixel_values,int(-size/2),int(size/2+0.5))
            for i in range(int(-size/2),int(size/2+0.5)):
                for j in range(int(-size/2),int(size/2+0.5)):
                    neighbor_row = row+i
                    neighbor_col = col+j
                    if neighbor_row>=0 and neighbor_col>=0 \
                            and neighbor_row < rows and neighbor_col < cols:
                        sum += img[neighbor_row,neighbor_col]
            new_img[row,col] = sum//(size**2)
    if bShow:
        cv.imshow("src", img)
        cv.imshow("mean filter img", new_img)
        cv.waitKey(0)
    return new_img

def meanFilter_demo():
    """
    Demo for mean filtering
    :return:
    """
    img = cv.imread("messi5.jpg", cv.IMREAD_GRAYSCALE)
    noised_img = salt_and_pepper_noise(img)
    meanFilter(noised_img, size=3, bShow=True)

def edgeDetection_demo():
    """
    Demo for edge detection
    :return:
    """
    img = cv.imread("messi5.jpg")
    #img = meanFilter(img, size=3) #Lam mo anh
    cv.imshow("src",img)
    edgeKernel1 = np.array([[ 1,  0, -1],
                            [ 0,  0,  0],
                            [-1,  0,  1]])
    edgeKernel2 = np.array([[ 1,  0, -1],
                            [ 0,  0,  0],
                            [-1,  0,  1]])
    edgeKernel3 = np.array([[-1, -1, -1],
                            [-1,  8, -1],
                            [-1, -1, -1]])
    edge1=cv.filter2D(src=img,ddepth=-1,kernel=edgeKernel1)
    edge2=cv.filter2D(src=img,ddepth=-1,kernel=edgeKernel2)
    edge3=cv.filter2D(src=img,ddepth=-1,kernel=edgeKernel3)
    cv.imshow("edge1",edge1)
    cv.imshow("edge2",edge2)
    cv.imshow("edge3",edge3)
    cv.waitKey(0)

def sharpenDemo():
    """
    Demo for sharpening image
    :return:
    """
    img = cv.imread("messi5.jpg")
    # img = meanFilter(img, size=3) #Lam mo anh
    cv.imshow("src", img)
    sharpKernel = np.array([[ 0, -1,  0],
                            [-1,  5, -1],
                            [ 0, -1,  0]])
    sharpImg = cv.filter2D(img,ddepth=-1,kernel=sharpKernel)
    cv.imshow("Sharpen image",sharpImg)
    cv.waitKey(0)

def gaussianBlurDemo():
    """
    Demo for gaussian blur
    :return:
    """
    img = cv.imread("messi5.jpg")
    # img = meanFilter(img, size=3) #Lam mo anh
    cv.imshow("src", img)
    gauKernel = np.array([[ 1, 2,  1],
                            [ 2,  4, 2],
                            [ 1, 2,  1]])/16
    gauImg = cv.filter2D(img,ddepth=-1,kernel=gauKernel)
    cv.imshow("Gaussian blurS image",gauImg)
    cv.waitKey(0)

def corelation(img1,img2,bShow=False):
    """
    calculate correlation between 2 images
    return the correlation value
    :param img1:
    :param img2:
    :param bShow:
    :return:
    """
    # Apply template Matching
    res = cv.matchTemplate(img1,img2,cv.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    return max_val

def correlationDemo():
    """
    demo for correlation
    :return:
    """
    img = cv.imread("messi5.jpg", cv.IMREAD_GRAYSCALE)
    noised_img = salt_and_pepper_noise(img)
    median_img = medianFilter(noised_img, size=3)
    print("Correlation between src and src:", corelation(img, img))
    print("Correlation between src and salt-pepper image:",corelation(img,noised_img))
    print("Correlation between src and median filter image:", corelation(img, median_img))
    cv.imshow("src",img)
    cv.imshow("noise img",noised_img)
    cv.imshow("median img",median_img)
    cv.waitKey(0)


def bilateralFilter(img,size=15,sigmaColor=75,sigmaSpace=75,bShow=False):
    """
    use bilateral filter to preserve edge info + filter noise
    :param img: img to be filtered
    :param size: kernel size
    :param sigmaColor: the smaller sigma, the sharper edge
    :param sigmaSpace: the smaller sigmaSpace, the finner image( filter more)
    :return: denoise image
    """

    # Apply bilateral filter with d = 15,
    # sigmaColor = sigmaSpace = 75.
    bilateral = cv.bilateralFilter(img, size, sigmaColor, sigmaSpace)
    if bShow:
        cv.imshow("src",img)
        cv.imshow("bilateral",bilateral)
        cv.waitKey(0)
    return bilateral

def bilateralDemo():
    """
    Demo for bilateral filter
    :return:
    """
    img = cv.imread("taj.jpg")
    #noised_img = meanFilter(img,7)
    bilateral = bilateralFilter(img)
    median = cv.medianBlur(img,3) # median filter
    mean = cv.blur(img,(3,3)) #mean filter
    cv.imshow("src",img)
    cv.imshow("bilateral",bilateral)
    cv.imshow("median",median)
    cv.imshow("mean",mean)
    cv.waitKey(0)

def main():
    #medianFilter_demo()
    #meanFilter_demo()
    #edgeDetection_demo()
    #sharpenDemo()
    #gaussianBlurDemo()
    #correlationDemo()
    bilateralDemo()

main()
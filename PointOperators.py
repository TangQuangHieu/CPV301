import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
def changeImageColor(img,alpha,beta,bShow=False):
    """
    out = alpha*img+beta
    :param img: input image ( gray scale)
    :param alpha:  >0 and <1
    :param beta: offset value
    :return: out image
    """
    assert len(img.shape)==2," Must be a gray scale image"
    out = np.array(img.shape,np.uint8)
    out = img*alpha+beta
    if bShow:
        cv.imshow("src",img)
        cv.imshow("After changing color",out)
        cv.waitKey(0)
    return out

def blendImages(img1,img2,alpha,bShow=False):
    """
    mix 2 image by alpha
    :param img1: input1
    :param img2: input2
    :param alpha: blend param
    :return: alpha*img1+(1-alpha)*img2
    """
    out = np.zeros(img1.shape,np.float64)
    if img1.shape[0] != img2.shape[0] or\
            img1.shape[1] != img2.shape[1]:
        img2 = cv.resize(img2,(img1.shape[1],img1.shape[0]))
    out = np.uint8(img1*alpha+img2*(1.-alpha))
    if bShow:
        cv.imshow("src1",img1)
        cv.imshow("src2",img2)
        cv.imshow("blended imgs",out)
        cv.waitKey(0)
    return out
def drawGammaCorrectionLine():
    """
    Drawing several line of gamma correction for input [0,255]
    :return:
    """
    x = np.linspace(0,255,256)
    gamma_list = [0.25,0.5,1,1.5,2.2,2.8]
    fig, ax = plt.subplots()

    colors = ['b','g','r','c','m','y','k','black']
    legends=[]
    for i,gamma in enumerate(gamma_list):
        ci = i%len(colors)
        y = ((x/255.)**(1./gamma))*255.
        plt.plot(x,y,color=colors[ci],linewidth=2,label=gamma)
        legends.append("gamma="+str(gamma))
    #Deco for the image
    ax.legend(legends)
    ax.set_xlabel("intensity")
    ax.set_ylabel("gamma correction")
    ax.set_title("Gamma correction map")
    ax.set_ylim([0,256])
    ax.set_xlim([0,256])
    plt.show()

def gammaCorrection(img,gamma,bShow=False):
    """
    Do gamma correction algo
    :param img: input image
    :param gamma: correction param
    :return: gamma correction image
    """
    color_map = [int(((i/255.)**(1./gamma))*255. ) for i in range(256)]
    print(color_map)
    new_img = np.zeros(img.shape,np.uint8)
    for row in range((img.shape[0])):
        for col in range(img.shape[1]):
            new_img[row,col] = (color_map[img[row,col][0]],
                                          color_map[img[row,col][1]],
                                                    color_map[img[row,col][2]])
    if bShow:
        cv.imshow("src",img)
        cv.imshow("gamma correction",new_img)
        cv.waitKey(0)
    return new_img

def drawHistogram(img,title='src'):
    """
    Draw histogram of given img
    :param img:
    :return: draw histogram of the img
    """
    cv.imshow(title, img)
    cv.waitKey(0)
    fig, ax = plt.subplots()
    x = np.linspace(0,255,256)
    y = np.zeros(x.shape,np.float64)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            y[img[row,col]]+=1
    cum_hist = np.zeros(y.shape,np.float64)
    cum_hist[0]=y[0]
    for i in range(1,len(cum_hist)):
        cum_hist[i]=cum_hist[i-1]+y[i]
    max_hist = np.float64(np.max(y))
    max_cum_hist = np.float64(np.max(cum_hist))
    # Nomarlization for drawing
    y *= (255./max_hist)
    cum_hist *=(255./max_cum_hist)
    plt.bar(x,y,color='b',label="histogram")
    plt.plot(x,cum_hist,color='r',label="Cummulate histogram")
    plt.legend(["Cummulate histogram","histogram"])
    ax.set_xlim(0,256)
    ax.set_ylim(0,256)
    ax.set_title("Histogram "+title)
    #plt.show()
    plt.pause(0.01)


def histogram_equalization(img,bShow=False):
    """
    :param img: input image (gray scale)
    :param bShow: whether to show image or not
    :return: equalised histogram image
    """
    assert len(img.shape)==2,"Must be a gray image"
    #1.Calculate hist of image
    hist = np.zeros(256,np.int16)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            hist[img[row,col]]+=1
    #print(hist)
    #2.Caculate cummulate sum his of the image
    cum_hist = np.zeros(256,np.int64)
    cum_hist[0]=hist[0]
    for i in range(1,len(hist)):
        cum_hist[i]=cum_hist[i-1]+hist[i]

    max_cum_hist = np.max(cum_hist)
    min_cum_hist = np.min(cum_hist)
    print(cum_hist)
    #3.create map for old pixel values
    hist_map = np.zeros(256, np.uint8)
    for i in range(len(cum_hist)):
        hist_map[i]=np.uint8((cum_hist[i]-min_cum_hist)/(max_cum_hist-min_cum_hist)*255.)

    #4.create histogram equalised image
    new_img = np.zeros(img.shape,np.uint8)
    for row in range(new_img.shape[0]):  # traverse by row (y-axis)
        for col in range(new_img.shape[1]):  # traverse by column (x-axis)
            new_img[row, col] = hist_map[img[row, col]]

    if bShow:
        cv.imshow("source",img)
        cv.imshow("Histogram equalization",new_img)
        cv.waitKey(0)
    return new_img

def main():
    #messi = cv.imread("messi5.jpg")

    #sudoku = cv.imread("sudoku.jpg")
    #blendImages(messi,sudoku,0.5,True)
    #gammaCorrection(messi,2.2,True)
    drawGammaCorrectionLine()
    #gray = cv.cvtColor(messi, cv.COLOR_BGR2GRAY)
    #print(gray.shape)
    #over_expose = changeImageColor(gray,1,50,False) # img_new = alpha*img+beta
    #drawHistogram(over_expose, title="over expose")
    #equal_hist = histogram_equalization(over_expose,False)
    #drawHistogram(equal_hist,title="equalization")
    #plt.show()

main()




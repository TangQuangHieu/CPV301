'''
Stitching sample
================

Show how to use Stitcher API from python in a simple way to stitch panoramas
or scans.
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import sys



"""parser = argparse.ArgumentParser(prog='stitching.py', description='Stitching sample.')
parser.add_argument('--mode',
                    type=int, choices=modes, default=cv.Stitcher_PANORAMA,
                    help='Determines configuration of stitcher. The default is `PANORAMA` (%d), '
                         'mode suitable for creating photo panoramas. Option `SCANS` (%d) is suitable '
                         'for stitching materials under affine transformation, such as scans.' % modes)
parser.add_argument('--output', default='result.jpg',
                    help='Resulting image. The default is `result.jpg`.')
parser.add_argument('img', nargs='+', help='input images')

__doc__ += '\n' + parser.format_help()"""

def display_video(path):
    cap = cv.VideoCapture(path)
    frame,ret = cap.read()
    cntFrame=0
    while 1:
        ret,frame = cap.read()

        if ret:
            cv.imwrite("./stitch_images/img{:04d}.jpg".format(cntFrame), frame)
            cv.waitKey(10)
            cntFrame+=1
            cv.imshow("video",frame)
            k = cv.waitKey(20)
            if k%256 == 32:
                #Space to break
                break
        else: break
    cap.release()
    return cntFrame


#display_video("./stitching_video.mp4")
def main():
    cntFrame = display_video("./stitching_video.mp4")
    modes = (cv.Stitcher_PANORAMA, cv.Stitcher_SCANS)
    # read input images
    idxs = np.linspace(0,cntFrame,50).astype('int')
    #print(idxs)

    imgs = []
    for idx in idxs:
        img = cv.imread("stitch_images/img{:04d}.jpg".format(idx))
        imgs.append(img)
    shape = idxs.reshape(-1,5).shape
    _, axs = plt.subplots(shape[0], shape[1], figsize=(12, 12))
    axs = axs.flatten()
    for img, ax in zip(imgs, axs):
        ax.imshow(img)
    plt.show()

    stitcher = cv.Stitcher.create(modes[1])
    status, pano = stitcher.stitch(imgs)
    if status != cv.Stitcher_OK:
        print("Can't stitch images, error code = %d" % status)
        sys.exit(-1)
    cv.imwrite("stitch_images/panorama.jpg", pano)
    print("stitching completed successfully. %s saved!" % modes[1])
    cv.waitKey(10)
    cv.imshow("stitch_image",pano)
    cv.waitKey(0)
    print('Done')


if __name__ == '__main__':
    #print(__doc__)
    main()
    cv.destroyAllWindows()

main()
print("Hello WOrld")


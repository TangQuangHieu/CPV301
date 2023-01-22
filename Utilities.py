import cv2 as cv
import numpy as np
def rotatePt(pt,centerPt,angle):
    """
    :param pt: point to be rotated, tuple
    :param centerPt: anchor point
    :param angle: in degree
    :return: coordinates after the rotating
    """
    angle = angle*np.pi/180
    M = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]).astype('float')
    X = np.array([pt[0]-centerPt[0],pt[1]-centerPt[1]]).astype('float')
    X_new = np.add(M.dot(X),centerPt)
    return tuple(X_new.astype('int'))

def getUserChoice(options):
    """
    :param options: list of options
    :return: a number indicate which option is chosen
    """
    for i in range(len(options)):
        print(i+1,":",options[i])
    return int(input("Please choose (1.."+str(len(options))+"):"))

def drawRectangle(rec,img):
    """
    :param cvRectangle: instance to be drawn
    :param img: canvas
    :return: img with new rectangle
    """
    #Green color
    color = (0, 0, 255)
    topLPt = rotatePt((rec.cx-rec.w/2,rec.cy-rec.h/2),(rec.cx,rec.cy),rec.angle)
    topRPt = rotatePt((rec.cx+rec.w/2,rec.cy-rec.h/2),(rec.cx,rec.cy),rec.angle)
    botRPt = rotatePt((rec.cx + rec.w / 2, rec.cy + rec.h / 2), (rec.cx, rec.cy), rec.angle)
    botLPt = rotatePt((rec.cx - rec.w / 2, rec.cy + rec.h / 2), (rec.cx, rec.cy), rec.angle)
    #img = cv.line(img,topLPt,topRPt,color,2)
    #img = cv.line(img, topRPt, botRPt, color, 2)
    #img = cv.line(img, botRPt, botLPt, color, 2)
    #img = cv.line(img, botLPt, topLPt, color, 2)
    pts = np.array([topLPt,topRPt,botRPt,botLPt], np.int32)
    img = cv.fillPoly(img, pts = [pts], color =color)
    return img.astype('int')

class cvRectangle:
    def __init__(self):
        self.cx=0
        self.cy=0
        self.w=0
        self.h=0
        self.angle=0
    def rotate(self,angle):
        self.angle=angle
    def translate(self,offset):
        self.cx+=offset[0]
        self.cy+=offset[1]
    def scale(self,scales):
        self.w *= scales[0]
        self.h *= scales[1]


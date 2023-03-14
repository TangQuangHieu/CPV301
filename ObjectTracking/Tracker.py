import cv2 as cv
import numpy as np

class Tracker:
    common_id = 0
    def __init__(self,x,y,w,h):
        self.id = Tracker.common_id
        Tracker.common_id+=1
        self.age=0
        self.miss_detect = 0
        self.x=x
        self.y=y
        self.w=w
        self.h=h
        self.cost = 0
        self.history=[(x,y,w,h,self.cost)]
    def update(self,x,y,w,h):
        self.history.append = [(self.x, self.y, self.w, self.h,self.cost)]
        self.x = x
        self.y=y
        self.w=w
        self.h=h
        self.miss_detect = 0
    def predict(self):
        self.age+=1
        self.miss_detect+=1
    def cost(self,x,y,w,h):
        del_x = x-self.x
        del_y = y-self.y
        del_w = w-self.w
        del_h = h-self.h
        return np.sqrt(del_x**2+del_y**2+del_w**2+del_h**2)
    def isDead(self)->bool:
        return self.miss_detect==3





import cv2 as cv
import numpy as np
from scipy.optimize import linear_sum_assignment
class Association:
    def __init__(self):
        self.association_map={} #Track ID: box ID
        self.history=[]
    def association(self,trackers,boxes):
        size = max(len(trackers),len(boxes))
        cost_matrix = []
        for tracker in trackers:
            for box in boxes:
                cost_matrix.append(tracker.cost(box[0],box[1],box[2],box[3]))
            if(len(boxes)<len(trackers)):
                delta = len(trackers) - len(boxes)
                for i in range(delta):
                    cost_matrix.append(1000000)
        if len(trackers)<len(boxes):
            delta = len(boxes)-len(trackers)
            for i in range(delta):
                for box in boxes:
                    cost_matrix.append(1000000)
        n = np.max(len(trackers),len(boxes))
        cost_matrix = np.array(cost_matrix).reshape((n,n))
        row_ind,col_ind = linear_sum_assignment(cost_matrix)









import cv2 as cv
import numpy as np

class harr_detector:
    def __init__(self):
        self.face_casecade = cv.CascadeClassifier()
        self.eye_casecade = cv.CascadeClassifier()
        self.face_casecade.load("./data/haarcascades/haarcascade_frontalface_alt.xml")
        self.eye_casecade.load("./data/haarcascades/haarcascade_eye_tree_eyeglasses.xml")
    def detect_face(self,img)->list:
        """
        trả về box cho các khuôn mặt trong hình
        :param img:
        :return:
        """
        img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        img_gray = cv.equalizeHist(img_gray)
        return self.face_casecade.detectMultiScale(img_gray)
    def detect_eyes(self,face_img)->list:
        """
        Trả về box cho các mắt trong hình
        :param img:
        :return:
        """
        img_gray = cv.cvtColor(face_img, cv.COLOR_BGR2GRAY)
        img_gray = cv.equalizeHist(img_gray)
        return self.eye_casecade.detectMultiScale(img_gray)

    #Các hàm vẽ
    def draw_box(self,img,boxes):
        """
        Vẽ box lên hình
        :param img:
        :param boxes:(x,y,w,h)
        :return:
        """
        for box in boxes:
            #b = list(box)
            #print(b)
            #print(box[0],";",box[1],",",box[2])
            (x, y, w, h) = box

            center=(x+w//2,y+h//2)
            img = cv.ellipse(img,center,(w//2,h//2),0,0,360,(255,0,255),4)
        return img

def main():
    harr = harr_detector()

    cap = cv.VideoCapture(0)
    while True:
        ret,frame = cap.read()
        if ret:
            face_boxes = harr.detect_face(frame)
            eye_boxess=[]
            for box in face_boxes:
                (x,y,w,h)=box
                roi = frame[y:y+h,x:x+w]
                cv.imshow("face",roi)
                cv.waitKey(5)
                eye_boxes = harr.detect_eyes(roi)
                if len(eye_boxes)>0:
                    #print(eye_boxes)
                    for eye_box in eye_boxes:
                        eye_box[0]+=x
                        eye_box[1]+=y
                        eye_boxess.append(eye_box)
                    #print(eye_boxess)
            #print(eye_boxess)
            #eye_boxess = np.array(eye_boxess)
            #eye_boxess = eye_boxess.ravel().tolist()
            #print(eye_boxess)
            if len(face_boxes)>0: frame = harr.draw_box(frame,face_boxes)
            if len(eye_boxess)>0: frame = harr.draw_box(frame, eye_boxess)
            cv.imshow("video",frame)
            if cv.waitKey(10) == 27:break
    img = cv.imread("family.jpg")
    img = cv.resize(img,(500,500))
    face_boxes = harr.detect_face(img)
    harr.draw_box(img, face_boxes)
    cv.imshow("img", img)
    cv.waitKey(0)

main()





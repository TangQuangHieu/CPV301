import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import time
#Tham khảo: https://docs.opencv.org/3.4/da/d60/tutorial_face_main.html
class EigenFace:
    def __init__(self):
        self.imgs = []
        self.labels=[]
        self.trainImgs=[]
        self.trainLabels=[]
        self.testImgs=[]
        self.testLabels=[]
        self.path = None
        self.num_components = 10
        self.threshold = 10.0
        self.eigenModel = cv.face.EigenFaceRecognizer.create(self.num_components,self.threshold)
        #print("Finish initilization")
        #print(self.eigenModel)
    def norm_0_255(self,img):
        """
        Chuyển về khoảng [0,255]
        :param img:
        :return:
        """
        assert(len(img.shape)==2)
        c=1
        dest = np.zeros_like(img)
        if c==1:
            return cv.normalize(img,dest,0,255,cv.NORM_MINMAX,cv.CV_8UC1)
        elif c==3:
            return cv.normalize(img,dest,0,255,cv.NORM_MINMAX,cv.CV_8UC3)
        else: return img
    def read_csv(self,filename,separator=";"):
        """
        Đọc từng dòng trong file csv theo dạng sau:
        ./at/s1/1.pgm;0. Trong đó số 1 là đường dẫn,
        số 2 là class label
        :param filename: Tên file
        :param separator: ký hiệu giữa 2 data trong 1 dòng
        :return: trả về hình và label của hình dạng 2 list
        """
        self.path=filename
        self.imgs=[]
        self.labels=[]
        with open(self.path,"r") as f_in:
            lines = f_in.readlines()

            for line in lines:
                #print(line)
                tokens = line.split(';')
                #print(tokens)
                if len(tokens)==2:

                    self.imgs.append(self.norm_0_255(cv.imread(tokens[0],cv.IMREAD_GRAYSCALE)))
                    self.labels.append(int(tokens[1]))
        self.labels = np.array(self.labels)
        return self.imgs,self.labels

    def train_test_split(self,ratio=0.8):
        """
        Mặc định là 80% data cho train, 20% cho
        test
        :param ratio:
        :return:
        """
        rnd.seed(33)
        idxs = np.arange(0,len(self.labels))
        rnd.shuffle(idxs)
        self.trainImgs = self.imgs[0:int(ratio*len(self.imgs))]
        self.trainLabels = self.labels[0:int(ratio*len(self.labels))]
        self.testImgs = self.imgs[int((1.-ratio)*len(self.imgs)):]
        self.testLabels = self.labels[int((1.-ratio)*len(self.labels)):]
    def train(self):
        """
        Chỉ train eigenModel với train image & train labels
        :return:
        """
        #print(self.trainLabels)
        self.eigenModel.train(self.trainImgs,self.trainLabels)
        self.predict_train = []
        print("***Training predict result***")
        for img in self.trainImgs:
            #print(img.shape)
            label,conf=self.eigenModel.predict(img)
            print("label:",label,"Confident:",conf)
            self.predict_train.append(label)
        #Tính toán các giá trị của train
        diff = self.predict_train == self.trainLabels
        #print(diff)
        print("===================================")
        print("Train Accuracy:",diff.sum()/len(self.trainLabels)*100,"%")
    def test(self):
        """
        Kiểm tra với tập kiểm tra
        :return:
        """
        self.predict_test = []
        for img in self.testImgs:
            label,conf= self.eigenModel.predict(img)
            self.predict_test.append(label)
        # Tính toán các giá trị của train

        diff = self.predict_test == self.testLabels
        #print(diff)
        print("Test Accuracy:", diff.sum() / len(self.testLabels) * 100, "%")

    def project_sub_space(self,img):
        """
        Trả về vector sau khi chiếu lên tọa độ của sub space
        :param img:
        :return:
        """
        #Bước 1. trừ đi mean của bộ data
        delta = img.reshape(img.shape[0]*img.shape[1],1) - self.eigenModel.getMean().reshape(-1,1)
        print(delta.shape)
        #bước 2. Tính x' theo công thức: y=WT(x−μ)
        return self.eigenModel.getEigenVectors().T.dot(delta)

    def reconstruct_img(self,img):
        """
        Chiếu hình lên subspace, sau đó tái tạo lại hình từ hình chiếu
        :param img:
        :return:
        """
        #Bước 1. Chiếu hình lên sub space
        y = self.project_sub_space(img)
        #Bước 2. Tái tạo dữ liệu từ hình chiếu theo công thức x=Wy+μ
        x = self.eigenModel.getEigenVectors().dot(y)+self.eigenModel.getMean().reshape(-1,1)
        return x.reshape(img.shape)

    #Các hàm vẽ, should be deployed in another class
    def display_mean_face(self):
        """
        Vẽ mặt và eigen face của nó
        :param img:
        :return:
        """
        #Bước 1: Lấy eigen vector
        mean_face = self.eigenModel.getMean().reshape(self.imgs[0].shape[0],self.imgs[0].shape[1])
        print("Mean face shape:",mean_face.shape)
        plt.imshow(mean_face,cmap='gray', vmin=0, vmax=255)
        plt.title("Mean face")
        plt.show()

    def display_eigen_faces(self):
        W=self.eigenModel.getEigenVectors()
        #Tìm cách sắp xếp hình hợp lý nhất
        num_figure_x=int(W.shape[1]/2)
        while W.shape[1] % num_figure_x != 0 and num_figure_x>1:
            num_figure_x-=1
        figure, axis = plt.subplots(int(W.shape[1]/num_figure_x), num_figure_x)
        for i in range(0,W.shape[1]):
            #print(i)
            #Lấy từng vector cột của W và reshape về cùng kích cỡ với ảnh gốc
            eigenFace = W[:,i].reshape(self.imgs[0].shape)
            axis[i//num_figure_x,i%num_figure_x].imshow(eigenFace,cmap = 'jet')
            axis[i // num_figure_x, i % num_figure_x].set_title("eigenFace{:04d}".format(i))
        plt.show()

    def display_reconstruction_image(self,img):
        reconstructed_img = self.reconstruct_img(img)
        figure, axis = plt.subplots(1, 2)
        axis[0].imshow(img,cmap='gray')
        axis[0].set_title("Src")
        axis[1].imshow(reconstructed_img,cmap='gray')
        axis[1].set_title("Dest")
        plt.show()
def main():
    ef = EigenFace()
    ef.read_csv("./at_mod.csv")
    ef.train_test_split()
    #ef.train()
    #eigenModel = cv.face.EigenFaceRecognizer.create(10, 10)
    #eigenModel.train(ef.trainImgs,ef.trainLabels)
    ef.train()
    ef.test()
    ef.display_mean_face()
    ef.display_eigen_faces()
    ef.display_reconstruction_image(ef.imgs[20])
main()





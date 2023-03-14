import cv2 as cv
import time
import getpass  # importing maskpass library

def rtsp_kbone_streaming(user_name="admin",password="1234", ip="192.168.229.129",port=554,channel=1,subtype=0):
    """
    :param user_name: tên account ( thường là admin)
    :param password:  safecode ( dán ở mặt sau của camera)
    :param ip: Địa chỉ ip hiện tại của camera
    :param port: cổng vào của camera ( thường là 554)
    :param channel: bắt đầu từ 1
    :param subtype: bắt đầu từ 0
    :return: trả về luồng streaming cho camera
    """
    url = "rtsp://{}:{}@{}:{}/cam/realmonitor?channel=1&subtype=0".format(user_name,password,ip,str(port),
                                                                          str(channel),str(subtype))
    cap = cv.VideoCapture(url)

    cv.waitKey(5)
    #Chờ khoảng 5 giây cho kết nối ổn định
    return cap

def display_streaming(cap):
    """
    hiển thị luồng streaming xem có chạy không
    :param cap:
    :return:
    """
    k=''
    t_mode = False
    while 1:
        ret,frame = cap.read()
        if ret:
            frame = cv.resize(frame,(640,480))
            cv.imshow("Video", frame)
            k = cv.waitKey(10) % 256
            if k==27:
                break #Kết thúc khi ấn ESC

            #print(k)
def main():
    #Code test rtsp
    user_name = input("Nhập account:")
    password = input("Nhập password:")
    #password = getpass.getpass("Nhập password:")
    ip = input("Nhập IP:")
    port = int(input("Nhập port"))
    cap = rtsp_kbone_streaming(user_name,password,ip,port)
    if cap == None:
        print("Không kết nối được camera!")
    else:
        display_streaming(cap)

main()


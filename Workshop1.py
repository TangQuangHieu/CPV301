from Utilities import *

drawing = False
ix,iy = -1,-1
img = np.ones((200, 200,3),np.uint8) * 255
rec = cvRectangle()

# define mouse callback function to draw circle
def draw_rectangle(event, x, y, flags, params):
   global drawing,ix, iy, img, rec
   if event == cv.EVENT_LBUTTONDOWN:
      #print("Button down")
      drawing = True
      ix = x
      iy = y
   elif event == cv.EVENT_MOUSEMOVE:
      #print("Mouse move")
      if drawing == True:
         cv.rectangle(img, (ix, iy), (x, y), (0, 0, 255), -1)
   elif event == cv.EVENT_LBUTTONUP:
      #print("Button up")
      drawing = False
      rec.w = abs(x - ix)
      rec.h = abs(y - iy)
      rec.cx = (x+ix)/2
      rec.cy = (y+iy)/2
      rec.angle=0
      cv.rectangle(img, (ix, iy), (x, y), (0, 0, 255), 2)

def main():
    global ix,iy,img,rec
    cv.namedWindow("image",0)
    cv.imshow("image",img)
    cv.setMouseCallback("image", draw_rectangle,[img,rec])

    options = ("Create a white background",
               "Draw rectangle",
               "Translation",
               "Rotation",
               "Scaling")
    while(True):
        # img = setColorForImg(img)
        userChoice = getUserChoice(options)
        if userChoice == 1:
            img = np.ones((200, 200,3),np.uint8) * 255
        elif userChoice == 2:
            #Draw rectangle here
            ix, iy = -1, -1
            while(True):
                cv.imshow("image", img)
                if cv.waitKey(10) == 27:
                    break
        elif userChoice == 3:
            dx = int(input("dx="))
            dy = int(input("dy="))
            rec.translate((dx,dy))
            img = np.ones((200, 200,3),np.uint8) * 255
            drawRectangle(rec,img)
        elif userChoice == 4:
            rec.rotate(int(input("Rotation angle = ")))
            img = np.ones((200, 200,3),np.uint8) * 255
            drawRectangle(rec,img)
        elif userChoice == 5:
            sx = float(input("sx="))
            sy = float(input("sy="))
            rec.scale((sx,sy))
            img = np.ones((200, 200,3),np.uint8) * 255
            drawRectangle(rec,img)
        else: break
        cv.imshow("image",img)
        cv.waitKey(20)

main()
from Utilities import *

pt = (0,10)
center =(0,0)
angles=[0,30,60,90,120,150,180]
for angle in angles:
    print("Angle",angle,"rotated pt:",rotatePt(pt,center,angle))
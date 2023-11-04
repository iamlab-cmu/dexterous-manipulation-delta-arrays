import numpy as np
import cv2
import matplotlib.pyplot as plt


# define a video capture object
vid = cv2.VideoCapture(0) 
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)

while(True):
    ret, frame = vid.read()
             

    if ret == True:
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
       
           
    else:
       
        print("ret is empty")
        break
 
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
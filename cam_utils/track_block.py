import numpy as np
import cv2
import pickle


cam = cv2.VideoCapture(1)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
cv2.namedWindow("test")
l_b=np.array([20, 100, 100])# lower hsv bound for red
u_b=np.array([33, 255, 255])# upper hsv bound to red
kernel = np.ones((11,11),np.uint8)

while True:
    ret, frame = cam.read()
    print(frame.shape)
    if not ret:
        print("Failed to Get WebCam img")
        break

    temp = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(temp, l_b, u_b)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours,_= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    max_contour = contours[0]
    for contour in contours:
        if cv2.contourArea(contour)>cv2.contourArea(max_contour):
            max_contour=contour
    if len(contours) != 0:
        # max_contour = max(contours, key = cv2.contourArea)
        hull = cv2.convexHull(max_contour)
        frame = cv2.resize(frame, (frame.shape[1]//4, frame.shape[0]//4))
        cv2.drawContours(frame, max_contour//4, -1, (0,255,0), 5)
        
        # c = max(contours, key = cv2.contourArea)
        # x,y,w,h = cv2.boundingRect(c)
        # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("frame",frame)
    # cv2.imshow("mask",mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
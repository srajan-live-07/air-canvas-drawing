import numpy as np
import cv2
from collections import deque


def setValues(x):
    print("")

# marker color points
cv2.namedWindow("color detectors")
cv2.createTrackbar("Upper Hue", "color detectors", 153, 180, setValues)
cv2.createTrackbar("Upper Saturation", "color detectors", 255, 255, setValues)
cv2.createTrackbar("Upper Value", "color detectors", 255, 255, setValues)

cv2.createTrackbar("Lower Hue", "color detectors", 64, 180, setValues)
cv2.createTrackbar("Lower Saturation", "color detectors", 72, 255, setValues)
cv2.createTrackbar("Lower Value", "color detectors", 49, 255, setValues)
  

bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

kernel = np.ones((5, 5), np.uint8)

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0

# Create a white canvas with uint8 dtype to avoid OpenCV warnings
paintWindow = np.zeros((471, 636, 3), np.uint8) + 255


paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), -1)
paintWindow = cv2.rectangle(paintWindow, (160, 1), (255, 65), colors[0], -1)
paintWindow = cv2.rectangle(paintWindow, (275, 1), (370, 65), colors[1], -1)
paintWindow = cv2.rectangle(paintWindow, (390, 1), (485, 65), colors[2], -1)
paintWindow = cv2.rectangle(paintWindow, (505, 1), (600, 65), colors[3], -1)

cv2.putText(paintWindow, "CLEAR ALL", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

cv2.namedWindow("Paint", cv2.WINDOW_AUTOSIZE)


#capture live webcam feed
cap = cv2.VideoCapture(0)

while True:
    Success,frame = cap.read()    
    frame=cv2.flip(frame,1)
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV) # hue saturation value
    
    u_hue = cv2.getTrackbarPos("Upper Hue", "color detectors")
    u_saturation = cv2.getTrackbarPos("Upper Saturation", "color detectors")
    u_value = cv2.getTrackbarPos("Upper Value", "color detectors")
    l_hue = cv2.getTrackbarPos("Lower Hue", "color detectors")
    l_saturation = cv2.getTrackbarPos("Lower Saturation", "color detectors")
    l_value = cv2.getTrackbarPos("Lower Value", "color detectors")

    upper_hsv = np.array([u_hue, u_saturation, u_value])
    lower_hsv = np.array([l_hue, l_saturation, l_value])
    
    frame = cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), -1)
    frame = cv2.rectangle(frame, (160, 1), (255, 65), colors[0], -1)
    frame = cv2.rectangle(frame, (275, 1), (370, 65), colors[1], -1)
    frame = cv2.rectangle(frame, (390, 1), (485, 65), colors[2], -1)
    frame = cv2.rectangle(frame, (505, 1), (600, 65), colors[3], -1)

    cv2.putText(frame, "CLEAR ALL", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    #creating mask 
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    

    #find contours
    cntz,z= cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None

    if len(cntz) > 0:
        cnt = sorted(cntz, key=cv2.contourArea, reverse=True)[0]
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        M = cv2.moments(cnt)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if center[1] <= 65:
            if 40 <= center[0] <= 140:  # Clear All
                bpoints = [deque(maxlen=1024)]
                gpoints = [deque(maxlen=1024)]
                rpoints = [deque(maxlen=1024)]
                ypoints = [deque(maxlen=1024)]

                blue_index = 0
                green_index = 0
                red_index = 0
                yellow_index = 0

                paintWindow[67:, :, :] = 255
            elif 160 <= center[0] <= 255:
                colorIndex = 0  # Blue
            elif 275 <= center[0] <= 370:
                colorIndex = 1  # Green
            elif 390 <= center[0] <= 485:
                colorIndex = 2  # Red
            elif 505 <= center[0] <= 600:
                colorIndex = 3  # Yellow
        else:
            if colorIndex == 0:
                bpoints[blue_index].appendleft(center)
            elif colorIndex == 1:
                gpoints[green_index].appendleft(center)
            elif colorIndex == 2:
                rpoints[red_index].appendleft(center)
            elif colorIndex == 3:
                ypoints[yellow_index].appendleft(center)
    else:
        bpoints.append(deque(maxlen=1024))
        blue_index += 1
        gpoints.append(deque(maxlen=1024))
        green_index += 1
        rpoints.append(deque(maxlen=1024))
        red_index += 1
        ypoints.append(deque(maxlen=1024))
        yellow_index += 1

    points= [bpoints, gpoints, rpoints, ypoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

    
    cv2.imshow("Live Feed", frame)
    cv2.imshow("White Window", paintWindow)
    cv2.imshow("mask",mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
        
cap.release()
cv2.destroyAllWindows()
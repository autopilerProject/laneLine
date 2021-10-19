import cv2
import numpy as np
from matplotlib import pyplot as plt, cm, colors

cap = cv2.VideoCapture('../project_video.mp4')

def birdsView(frame):
    img_size = (frame.shape[1], frame.shape[0])

    cv2.circle(frame, (590, 440), 5, (0, 0, 255), -1)
    cv2.circle(frame, (690, 440), 5, (0, 0, 255), -1)
    cv2.circle(frame, (200, 640), 5, (0, 0, 255), -1)
    cv2.circle(frame, (1000, 640), 5, (0, 0, 255), -1)

    cv2.circle(frame, (200, 0), 5, (0, 255, 0), -1)
    cv2.circle(frame, (1200, 0), 5, (0, 255, 0), -1)
    cv2.circle(frame, (200, 710), 5, (0, 255, 0), -1)
    cv2.circle(frame, (1200, 710), 5, (0, 255, 0), -1)

    src = np.float32([[590, 440],
                      [690, 440],
                      [200, 640],
                      [1000, 640]])

    # Window to be shown
    dst = np.float32([[200, 0],
                      [1200, 0],
                      [200, 710],
                      [1200, 710]])

    matrix = cv2.getPerspectiveTransform(src, dst)
    # Inverse matrix to unwarp the image for final window
    minv = cv2.getPerspectiveTransform(dst, src)
    birdseye = cv2.warpPerspective(frame, matrix, img_size)

    # Get the birdseye window dimensions
    height, width = birdseye.shape[:2]

    # Divide the birdseye view into 2 halves to separate left & right lanes
    birdseyeLeft  = birdseye[0:height, 0:width // 2]
    birdseyeRight = birdseye[0:height, width // 2:width]

    # Display birdseye view image
    # cv2.imshow("Birdseye" , birdseye)
    # cv2.imshow("Birdseye Left" , birdseyeLeft)
    # cv2.imshow("Birdseye Right", birdseyeRight)

    return birdseye, birdseyeLeft, birdseyeRight, minv


def processImage(inpImage):

    # Apply HLS color filtering to filter out white lane lines
    hls = cv2.cvtColor(inpImage, cv2.COLOR_BGR2HLS)
    lower_white = np.array([0, 160, 10])
    upper_white = np.array([255, 255, 255])
    mask = cv2.inRange(inpImage, lower_white, upper_white)
    hls_result = cv2.bitwise_and(inpImage, inpImage, mask = mask)

    # Convert image to grayscale, apply threshold, blur & extract edges
    gray = cv2.cvtColor(hls_result, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(thresh,(3, 3), 0)
    canny = cv2.Canny(blur, 40, 60)

    # Display the processed images
    # cv2.imshow("Image", inpImage)
    # cv2.imshow("HLS Filtered", hls_result)
    # cv2.imshow("Grayscale", gray)
    # cv2.imshow("Thresholded", thresh)
    # cv2.imshow("Blurred", blur)
    # cv2.imshow("Canny Edges", canny)

    return hls_result, gray, thresh, blur, canny




while True:
    _, frame = cap.read()

    birdView, birdViewL, birdViewR, minverse = birdsView(frame)

    hls, grayscale, thresh, blur, canny = processImage(birdView)
    hlsL, grayscaleL, threshL, blurL, cannyL = processImage(birdViewL)
    hlsR, grayscaleR, threshR, blurR, cannyR = processImage(birdViewR)

    cv2.imshow('result', frame)
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
import os
from scipy import optimize
from matplotlib import pyplot as plt, cm, colors


ym_per_pix = 30 / 720       # y 축 픽셀 당 거리
xm_per_pix = 3.7 / 720      # x 축 픽셀 당 거리


CWD_PATCH = os.getcwd()


def readVideo():
    inpImage = cv2.imread('solidWhiteCurve.jpg') #cv2.VideoCapture('project_video.mp4')


    return inpImage





def processImage(inpImage):
    # HLS 컬러 필터링으로 흰색 차선 검출
    hls = cv2.cvtColor(inpImage, cv2.COLOR_BGR2HLS)
    lower_white = np.array([0, 160, 10])
    upper_white = np.array([255, 255, 255])
    mask = cv2.inRange(inpImage, lower_white, upper_white)
    hls_result = cv2.bitwise_and(inpImage, inpImage, mask = mask)

    # 그레이 스케일로 변환, 한계값 적용, 블러처리하고 edge 추출
    gray = cv2.cvtColor(hls_result, cv2.COLOR_BAYER_BG2BGRA)
    ret, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(thresh, (3, 3), 0)
    canny = cv2.Canny(blur, 40, 60)

    cv2.imshow("Image", inpImage)
    cv2.imshow("HLS FIltered", hls_result)

processImage(readVideo())
import cv2 
import numpy as np
from numpy.lib.type_check import imag

image = cv2.imread('solidWhiteCurve.jpg')
mark = np.copy(image)   # image 복사

# BGR 제한 값 설정
blue_threshold = 200    # blue 한계점
green_threshold = 200   # green 한계점
red_threshold = 200     # red 한계점
bgr_threshold = [blue_threshold, green_threshold, red_threshold]    # bgr_threshold은 긱 색상을 배열로 갖는 2차원 배열 

# BGr 제한 값보다 작으면 검은색으로
thresholds = (image[:,:,0] < bgr_threshold[0]) \
    | (image[:,:,1] < bgr_threshold[1]) \
    | (image[:,:,2] < bgr_threshold[2])

mark[thresholds] = [0, 0, 0]

cv2.imshow('white', mark)   #흰색 추출 이미지 출력
cv2.imshow('result', image) # 이미지 출력
cv2.waitKey(0)


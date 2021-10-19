"""
흑백으로 바꾼 이유: 컬러 이미지가 오히려 많은 Edge 검출 + 정확도가 높음, 그래서 연산량도 많은 단점 -> 흑백으로 연산량 줄이기
Blur 효과: 이미지의 노이즈를 줄이고, 불필요한 gradient 없애기 위함

"""



import cv2  # opencv 사용
import numpy as np

def grayscale(img):  # 흑백이미지로 변환
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, low_threshold, high_threshold):  # Canny 알고리즘
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size): # 가우시안 필터
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

image = cv2.imread('solidWhiteCurve.jpg')   # 이미지 읽기
height, width = image.shape[:2] # 이미지 높이, 너비

gray_img = grayscale(image) # 흑백이미지로 변환

blur_img = gaussian_blur(gray_img, 3)   # Blur 효과, canny 전에 적용, 크기를 키울 수는 있지만 짝수는 불가능

canny_img = canny(blur_img, 70, 210)    # Canny edge 알고리즘, 비율 1:2 or 1:3

cv2.imshow('result', canny_img)         # Canny 이미지 출력
cv2.waitKey(0)
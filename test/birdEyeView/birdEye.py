import cv2
import numpy as np

cap = cv2.VideoCapture('../project_video.mp4')

while True:
    _, frame = cap.read()

    cv2.circle(frame, (590, 440), 5, (0, 0, 255), -1)
    cv2.circle(frame, (690, 440), 5, (0, 0, 255), -1)
    cv2.circle(frame, (200, 640), 5, (0, 0, 255), -1)
    cv2.circle(frame, (1000, 640), 5, (0, 0, 255), -1)

    cv2.circle(frame, (200, 0), 5, (0, 255, 0), -1)
    cv2.circle(frame, (1200, 0), 5, (0, 255, 0), -1)
    cv2.circle(frame, (200, 710), 5, (0, 255, 0), -1)
    cv2.circle(frame, (1200, 710), 5, (0, 255, 0), -1)

    # pts1 = np.float32([[255, 220], [355, 220], [155, 420], [455, 420]])
    # pts2 = np.float32([[0, 0],
    #                   [400, 0],
    #                   [0, 600],
    #                   [400, 600]])
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

    result = cv2.warpPerspective(frame, matrix, (1200, 710))

    cv2.imshow('result', frame)
    cv2.imshow('birdeye', result)
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
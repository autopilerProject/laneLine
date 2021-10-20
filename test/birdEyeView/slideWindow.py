import cv2
import numpy as np
from matplotlib import pyplot as plt, cm, colors

ym_per_pix = 30 / 720
# Standard lane width is 3.7 meters divided by lane width in pixels which is
# calculated to be approximately 720 pixels not to be confused with frame height
xm_per_pix = 3.7 / 720

cap = cv2.VideoCapture('../project_video.mp4')

def birdsView(frame):
    img_size = (frame.shape[1], frame.shape[0])

    # cv2.circle(frame, (590, 440), 5, (0, 0, 255), -1)
    # cv2.circle(frame, (690, 440), 5, (0, 0, 255), -1)
    # cv2.circle(frame, (200, 640), 5, (0, 0, 255), -1)
    # cv2.circle(frame, (1000, 640), 5, (0, 0, 255), -1)

    # cv2.circle(frame, (200, 0), 5, (0, 255, 0), -1)
    # cv2.circle(frame, (1200, 0), 5, (0, 255, 0), -1)
    # cv2.circle(frame, (200, 710), 5, (0, 255, 0), -1)
    # cv2.circle(frame, (1200, 710), 5, (0, 255, 0), -1)

    src = np.float32([[590, 440],
                      [700, 440],
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

def plotHistogram(inpImage):

    histogram = np.sum(inpImage[inpImage.shape[0] // 2:, :], axis = 0)

    midpoint = int(histogram.shape[0] / 2)
    leftxBase = np.argmax(histogram[:midpoint])
    rightxBase = np.argmax(histogram[midpoint:]) + midpoint

    plt.xlabel("Image X Coordinates")
    plt.ylabel("Number of White Pixels")

    # Return histogram and x-coordinates of left & right lanes to calculate
    # lane width in pixels
    return histogram, leftxBase, rightxBase

def slide_window_search(binary_warped, histogram):

    # Find the start of left and right lane lines using histogram info
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    midpoint = int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # A total of 9 windows will be used
    nwindows = 9
    window_height = int(binary_warped.shape[0] / nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 100
    minpix = 50
    left_lane_inds = []
    right_lane_inds = []

    #### START - Loop to iterate through windows and search for lane lines #####
    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
        (0,255,0), 2)
        cv2.rectangle(out_img, (win_xright_low,win_y_low), (win_xright_high,win_y_high),
        (0,255,0), 2)
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))
    #### END - Loop to iterate through windows and search for lane lines #######

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Apply 2nd degree polynomial fit to fit curves
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)


    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    ltx = np.trunc(left_fitx)
    rtx = np.trunc(right_fitx)
    plt.plot(right_fitx)
    # plt.show()

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # cv2.imshow('slideWindow', out_img)
    plt.plot(left_fitx,  ploty, color = 'yellow')
    plt.plot(right_fitx, ploty, color = 'yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

    return ploty, left_fit, right_fit, ltx, rtx, out_img



def general_search(binary_warped, left_fit, right_fit):

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
    left_fit[1]*nonzeroy + left_fit[2] + margin)))

    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
    right_fit[1]*nonzeroy + right_fit[2] + margin)))

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]


    ## VISUALIZATION ###########################################################

    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin,
                                  ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    cv2.fillPoly(window_img, np.int_([left_line_pts]), (100, 200, 100))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (100, 200, 100))
    # result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    # cv2.imshow('re', window_img)
    # cv2.imshow('re1', out_img)

    plt.plot(left_fitx,  ploty, color = 'yellow')
    plt.plot(right_fitx, ploty, color = 'yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

    ret = {}
    ret['leftx'] = leftx
    ret['rightx'] = rightx
    ret['left_fitx'] = left_fitx
    ret['right_fitx'] = right_fitx
    ret['ploty'] = ploty

    return ret, window_img


def draw_lane_lines(original_image, warped_image, Minv, draw_info):

    leftx = draw_info['leftx']
    rightx = draw_info['rightx']
    left_fitx = draw_info['left_fitx']
    right_fitx = draw_info['right_fitx']
    ploty = draw_info['ploty']

    warp_zero = np.zeros_like(warped_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    mean_x = np.mean((left_fitx, right_fitx), axis=0)
    pts_mean = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))])

    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    cv2.fillPoly(color_warp, np.int_([pts_mean]), (0, 255, 255))

    newwarp = cv2.warpPerspective(color_warp, Minv, (original_image.shape[1], original_image.shape[0]))
    # addh = cv2.hconcat([addh, newwarp])
    # cv2.imshow('ai view', addh2)
    result = cv2.addWeighted(original_image, 1, newwarp, 0.3, 0)

    return pts_mean, result, newwarp


def measure_lane_curvature(ploty, leftx, rightx):

    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

    # Choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Fit new polynomials to x, y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad  = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    # print(left_curverad, 'm', right_curverad, 'm')

    # Decide if it is a left or a right curve
    if leftx[0] - leftx[-1] > 60:
        curve_direction = 'Left Curve'
    elif leftx[-1] - leftx[0] > 60:
        curve_direction = 'Right Curve'
    else:
        curve_direction = 'Straight'

    return (left_curverad + right_curverad) / 2.0, curve_direction


def add_img(direction, frame):
    if direction == 'Left Curve':
        cv2.arrowedLine(frame, (frame.shape[1]//2+40, frame.shape[0]-100), (frame.shape[1]//2-40, frame.shape[0]-100), (0, 255, 0), thickness=3)
    elif direction == 'Right Curve':
        cv2.arrowedLine(frame, (frame.shape[1]//2-40, frame.shape[0]-100), (frame.shape[1]//2+40, frame.shape[0]-100), (0, 255, 0), thickness=3)
    else:
        cv2.arrowedLine(frame, (frame.shape[1]//2, frame.shape[0]-60), (frame.shape[1]//2, frame.shape[0]-140), (0, 255, 0), thickness=3)
    
    return frame
    


while True:
    _, frame = cap.read()

    birdView, birdViewL, birdViewR, minverse = birdsView(frame)

    hls, grayscale, thresh, blur, canny = processImage(birdView)
    hlsL, grayscaleL, threshL, blurL, cannyL = processImage(birdViewL)
    hlsR, grayscaleR, threshR, blurR, cannyR = processImage(birdViewR)

    
    hist, leftBase, rightBase = plotHistogram(thresh)
    # print(rightBase - leftBase)
    plt.plot(hist)
    # plt.show()


    ploty, left_fit, right_fit, left_fitx, right_fitx, window_out_img = slide_window_search(thresh, hist)

    draw_info, window_img = general_search(thresh, left_fit, right_fit)

    meanPts, result, newwarp = draw_lane_lines(frame, thresh, minverse, draw_info)

    curveRad, curveDir = measure_lane_curvature(ploty, left_fitx, right_fitx)

    result = add_img(curveDir, result)

    # deviation, directionDev = offCenter(meanPts, frame)

    window_img_r = cv2.resize(window_img, (window_img.shape[1]//3, window_img.shape[0]//3))
    birdView_r = cv2.resize(birdView, (birdView.shape[1]//3, birdView.shape[0]//3))
    window_out_img_r = cv2.resize(window_out_img, (window_out_img.shape[1]//3, window_out_img.shape[0]//3))
    newwarp_r = cv2.resize(newwarp, (newwarp.shape[1]//3, newwarp.shape[0]//3))

    birdView_r = cv2.line(birdView_r, (birdView_r.shape[1], 0), (birdView_r.shape[1], birdView_r.shape[0]), (255, 255, 255), 2)
    birdView_r = cv2.line(birdView_r, (0, birdView_r.shape[0]), (birdView_r.shape[1], birdView_r.shape[0]), (255, 255, 255), 2)

    window_out_img_r = cv2.line(window_out_img_r, (0, 0), (0, window_out_img_r.shape[0]), (255, 255, 255), 2)
    window_out_img_r = cv2.line(window_out_img_r, (0, window_out_img_r.shape[0]), (window_out_img_r.shape[1], window_out_img_r.shape[0]), (255, 255, 255), 2)
    
    window_img_r = cv2.line(window_img_r, (0, 0), (window_img_r.shape[1], 0), (255, 255, 255), 2)
    window_img_r = cv2.line(window_img_r, (window_img_r.shape[1], 0), (window_img_r.shape[1], window_img_r.shape[0]), (255, 255, 255), 2)

    newwarp_r = cv2.line(newwarp_r, (0, 0), (newwarp_r.shape[1], 0), (255, 255, 255), 2)
    newwarp_r = cv2.line(newwarp_r, (0, 0), (0, newwarp_r.shape[0]), (255, 255, 255), 2)
    

    addh1 = cv2.hconcat([birdView_r, window_out_img_r])
    addh2 = cv2.hconcat([window_img_r, newwarp_r])
    addv = cv2.vconcat([addh1, addh2])

    cv2.imshow('ai view', addv)

    cv2.imshow('result', result)
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
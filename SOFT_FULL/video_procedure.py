import numpy as np
import cv2
import geometry as geo
import frame_processing as fproc

def remove_dots(first_frame):
    gray_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    ret, edges = cv2.threshold(gray_frame, 10, 255, cv2.THRESH_BINARY)
    im2, contours, hierrarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour_idx in range(len(contours)):
        if cv2.contourArea(contours[contour_idx]) < 25:
            cv2.drawContours(edges,contours, contour_idx, 0,-1)
    return edges

def video_processing(filename):
    video = cv2.VideoCapture(filename)

    ret, frame = video.read()
    ret, frame = video.read()
    window_h, window_w = frame.shape[:2]
    thresholded_frame = remove_dots(frame.copy())
    cv2.imshow("bez tackica",thresholded_frame)
    line_x1, line_y1, line_x2, line_y2 = geo.find_line_coords(thresholded_frame.copy())
    main_line= geo.create_three_line_coords(line_x1, line_y1, line_x2, line_y2, window_w, window_h)

    video_speed = 10

    while video.isOpened():
        frame_openend, current_frame = video.read()

        if not frame_openend:
            break

        key = cv2.waitKey(video_speed)
        if key == 43:
            if video_speed - 2 < 0:
                video_speed = 1
            else:
                video_speed = video_speed - 2
        elif key == 45:
            video_speed = video_speed + 2
        elif key == 27:
            break
        elif key == 112:
            cv2.waitKey()

        fproc.filter_full_numers(filename, current_frame, main_line)


    video.release()
    cv2.destroyAllWindows()
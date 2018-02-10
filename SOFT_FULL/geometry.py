import numpy as np
import cv2
import frame_processing as fproc

def get_k(x1,y1,x2,y2):
    return (y2 - y1) / (x2 - x1)

def get_y_over_x(x1,y1,kinv,target_x):
    return kinv * (target_x - x1) + y1

def get_x_over_y(x1,y1,kinv,target_y):
    return (target_y - y1 + kinv*x1) / kinv

def find_line_coords(thresholded_frame):
    im2, contours, hierarchy = cv2.findContours(thresholded_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(thresholded_frame, contours, -1, 255, 3)
    min_line_length = 10000
    max_line_gap = 10

    lines = cv2.HoughLinesP(thresholded_frame, 1, np.pi / 180, 200, min_line_length, max_line_gap)
    return lines[0][0]

def create_three_line_coords(x1,y1,x2,y2,window_w,window_h):
    main_line = {'x1':x1, 'x2':x2, 'y1':y1, 'y2':y2}
    return main_line

def get_contour_extreme_points(contour):
    min_rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(min_rect)
    box = np.int0(box)
    return box

def get_leftmost_and_rightmost(main_line):
    if main_line['x1'] < main_line['x2']:
        leftmost_x = main_line['x1']
        leftmost_y = main_line['y1']
        rightmost_x = main_line['x2']
        rightmost_y = main_line['y2']
    else:
        leftmost_x = main_line['x2']
        leftmost_y = main_line['y2']
        rightmost_x = main_line['x1']
        rightmost_y = main_line['y1']

    return leftmost_x, leftmost_y, rightmost_x, rightmost_y

def is_contour_in_bounds(main_line, contour):
    k = get_k(main_line['x1'], main_line['y1'], main_line['x2'], main_line['y2'])
    kinv = -1 / k

    leftmost_x, leftmost_y, rightmost_x, rightmost_y = get_leftmost_and_rightmost(main_line)

    contour_points = get_contour_extreme_points(contour)
    for contour_point in contour_points:
        contour_x = contour_point[0]
        contour_y = contour_point[1]
        # levo:
        line_x = get_x_over_y(leftmost_x, leftmost_y, kinv, contour_y)
        if line_x > contour_x:
            return False
        # desno:
        line_x = get_x_over_y(rightmost_x, rightmost_y, kinv, contour_y)
        if line_x < contour_x:
            return False
    return True

def is_contour_about_to_intersect(main_line, contour):
    k = get_k(main_line['x1'], main_line['y1'], main_line['x2'], main_line['y2'])

    leftmost_x, leftmost_y, rightmost_x, rightmost_y = get_leftmost_and_rightmost(main_line)
    contour_points = get_contour_extreme_points(contour)
    for contour_point in contour_points:
        contour_x = contour_point[0]
        contour_y = contour_point[1]
        line_y = get_y_over_x(leftmost_x, leftmost_y, k, contour_x)
        if abs(contour_y - line_y) < 4:
            return True
    return False

def get_contour_center_point(contour):
    x,y,w,h = cv2.boundingRect(contour)
    return tuple((x+int(w/2), y+int(h/2)))

def get_distance_between_two_points(pt1, pt2):
    return np.sqrt( (pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2 )

def get_closest_contours(max_distance, a_contour):
    closest_contours = []
    closest_contour_key = -1
    min_distance = 10000

    for item_key in fproc.items_in_image:
        item = fproc.items_in_image[item_key]
        item_contour = item['contour']
        item_contour_center = get_contour_center_point(item_contour)
        contour_center = get_contour_center_point(a_contour)
        distance = get_distance_between_two_points(item_contour_center, contour_center)
        if distance < max_distance:
            closest_contours.append(item_contour)
            if min_distance > distance:
                closest_contour_key = item_key
                min_distance = distance

    return closest_contours, closest_contour_key

def should_contour_be_added(main_line, contour):
    k = get_k(main_line['x1'], main_line['y1'], main_line['x2'], main_line['y2'])

    leftmost_x, leftmost_y, rightmost_x, rightmost_y = get_leftmost_and_rightmost(main_line)
    contour_points = get_contour_extreme_points(contour)
    top_point = contour_points[0]
    for contour_point_idx in range(1, len(contour_points)):
        new_y = contour_points[contour_point_idx][1]
        old_y = top_point[1]
        if new_y < old_y:
            top_point = contour_points[contour_point_idx]
    top_y = top_point[1]
    top_x = top_point[0]

    res_y = get_y_over_x(leftmost_x, leftmost_y, k, top_x)
    if (int(top_y - res_y) >= 10) and (int(top_y - res_y) <= 20):
        return True
    return False

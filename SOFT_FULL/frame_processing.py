import numpy as np
import cv2
import geometry as geo
import knn_procedure as knnproc

items_in_image = {}
item_id_counter = 1
sum_of_numbers = 0

def get_new_id():
    global item_id_counter
    id = item_id_counter
    item_id_counter = item_id_counter + 1
    return id

def reset_information():
    global items_in_image, item_id_counter, sum_of_numbers
    items_in_image = {}
    item_id_counter = 1
    sum_of_numbers = 0

def threshold_frame(frame):
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresholded_frame = cv2.threshold(grayscale_frame, 180, 255, cv2.THRESH_BINARY)
    dilate_kernel = np.ones((2, 2), np.uint8)
    thresholded_frame = cv2.morphologyEx(thresholded_frame, cv2.MORPH_DILATE, dilate_kernel)
    return thresholded_frame

def filter_full_numers(filename, current_frame, main_line):
    global items_in_image, sum_of_numbers

    main_pt1 = (main_line['x1'], main_line['y1'])
    main_pt2 = (main_line['x2'], main_line['y2'])

    thresholded_frame = threshold_frame(current_frame)
    im2, contours_current, hierrarchy_current = cv2.findContours(thresholded_frame,
                                                                 cv2.RETR_TREE,
                                                                 cv2.CHAIN_APPROX_SIMPLE)

    lone_current_contours = []
    for contour_idx in range(len(contours_current)):
        if hierrarchy_current[0, contour_idx, 3] == -1:
            lone_current_contours.append(contours_current[contour_idx])

    resulting_current = current_frame#cv2.cvtColor(thresholded_frame, cv2.COLOR_GRAY2BGR)

    new_items = {}
    if len(items_in_image) == 0:
        for contour in lone_current_contours:
            item = {}
            item['contour'] = contour
            item['id'] = get_new_id()
            item['in-lr-bounds'] = geo.is_contour_in_bounds(main_line, contour)
            item['is-about-to-intersect'] = geo.is_contour_about_to_intersect(main_line, contour)
            item['has-intersected'] = False
            item['should-be-added'] = False
            item['was-ever-added'] = False
            new_items[item['id']] = item
        items_in_image = new_items.copy()
    else:
        for contour in lone_current_contours:
            close_items, closest_item_key = geo.get_closest_contours(20, contour)
            if len(close_items) != 0:
                item = items_in_image[closest_item_key]
                item['contour'] = contour
                item['in-lr-bounds'] = geo.is_contour_in_bounds(main_line, contour)
                has_intersected = item['is-about-to-intersect']
                item['is-about-to-intersect'] = geo.is_contour_about_to_intersect(main_line, contour)
                if (item['is-about-to-intersect'] is not True) and (has_intersected is True) and (item['in-lr-bounds'] is True):
                    item['has-intersected'] = True
                if item['has-intersected']:
                    item['should-be-added'] = geo.should_contour_be_added(main_line, contour)
                new_items[item['id']] = item
            else:
                item = {}
                item['id'] = get_new_id()
                item['contour'] = contour
                item['in-lr-bounds'] = geo.is_contour_in_bounds(main_line, contour)
                item['is-about-to-intersect'] = geo.is_contour_about_to_intersect(main_line, contour)
                item['has-intersected'] = False
                item['should-be-added'] = False
                item['was-ever-added'] = False
                new_items[item['id']] = item
        items_in_image = new_items.copy()

    for item_key in items_in_image:
        contour = items_in_image[item_key]['contour']
        x,y,w,h = cv2.boundingRect(contour)
        if items_in_image[item_key]['is-about-to-intersect'] and items_in_image[item_key]['in-lr-bounds']:
            cv2.rectangle(resulting_current, (x,y), (x+w,y+h), (0,255,0), 1)
        else:
            cv2.rectangle(resulting_current, (x, y), (x + w, y + h), (0, 0, 255), 1)
        if items_in_image[item_key]['has-intersected']:
            cv2.rectangle(resulting_current, (x,y), (x+w, y+h), (0,255,255), 1)
        if items_in_image[item_key]['should-be-added'] and items_in_image[item_key]['was-ever-added'] is not True:
            cv2.rectangle(resulting_current, (x,y), (x+w, y+h), (255,0,255), 2)
            items_in_image[item_key]['was-ever-added'] = True
            digit = knnproc.recognize_contour(items_in_image[item_key]['contour'], thresholded_frame)
            sum_of_numbers = sum_of_numbers + digit
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(resulting_current, str(item_key), (x + w, y + h), font, 0.5, (255, 255, 255), 1, cv2.LINE_4)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(resulting_current, "Suma: " + str(sum_of_numbers), (20, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_4)
    cv2.line(resulting_current, main_pt1, main_pt2, (0,0,255), 1)

    cv2.imshow(filename, resulting_current)


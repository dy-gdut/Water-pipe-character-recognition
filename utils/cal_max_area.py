import cv2


def cal_max_area(out_cv):
    contours, _ = cv2.findContours(out_cv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    for contour in contours:
        current_area = cv2.contourArea(contour)
        if current_area > max_area:
            max_area = current_area
    # print("max_area:{}".format(max_area))
    return max_area

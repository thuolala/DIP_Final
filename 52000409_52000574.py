# ---------- 1. LOAD LIBRABRY ----------
import os
import cv2 as cv
import numpy as np 
from skimage.morphology import skeletonize

# ---------- 2. READ INPUT IMAGES FROM FOLDER ----------
def read_from_path(path):
    images = []
    for filename in os.listdir(path):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            img_path = os.path.join(path, filename)
            img = cv.imread(img_path)
            if img is not None:
                images.append(img)
    return images

# Specify the path to read input images
path = os.getcwd() + "/input"
images = read_from_path(path)

# ---------- 3. SHOW IMAGES ----------
def show(img_list):
    i = 1
    for img in img_list: 
        cv.imshow('Image ' + str(i), img)
        i = i + 1
    cv.waitKey(0)
    cv.destroyAllWindows()

# ---------- 4. SAVE IMAGES ----------
def write_to_path(images_list):
    count = 1 
    path = os.getcwd() + "/output/"
    for img in images_list: 
        cv.imwrite(path + 'output' + str(count) + '.png', img)
        count += 1 

def write_to_path_precessing(path, images_list, step):
    count = 1 
    for img in images_list: 
        cv.imwrite(path + step + str(count) + '.png', img)
        count += 1 

# ---------- 5. PRE PROCESSING ---------- 
def pre_processing(img_list): 
    img_gray = [] 
    img_pre = [] 
    for img in img_list:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ####
        img_gray.append(gray)
        ####
        thresh = cv.threshold(gray, img.mean(), 255, cv.THRESH_BINARY)[1]
        thresh = 255 - thresh

        img_pre.append(thresh) 

    return img_gray, img_pre
images_gray, images_pre = pre_processing(images)

# ---------- 6. FIND CLOCK HAND (HOUR, MINUTE, SECOND) ----------  
# Sort contours by area 
def sort_by_area(li):
    return li[1]

# Find angle of each hand --> for get exact time 
def find_angle(hand,center):
    x_h = hand[0]
    y_h = hand[1]
    x_c = center[0]
    y_c = center[1]

    x_diff = x_h - x_c
    y_diff = y_h - y_c
    x_diff = float(x_diff)
    y_diff = float(y_diff)

    if(x_diff*y_diff > 0):
        if(x_diff >= 0 and y_diff > 0):
            return np.pi-np.arctan(x_diff/y_diff)
        elif(x_diff <= 0 and y_diff < 0):
            return 2*np.pi - np.arctan(x_diff/y_diff)
    elif(x_diff*y_diff < 0):
        if(y_diff >= 0 and x_diff < 0):
            return (3*np.pi)/4 + np.arctan(x_diff/y_diff)
        elif(y_diff <= 0 and x_diff > 0):
            return -np.arctan(x_diff/y_diff)

# Find hour by angle 
def get_hour(angle):
    hour = angle//30
    if(hour == 0):
        return 12
    else:
        return int(hour)

# Find min by angle
def get_min(angle):
    min = angle/(np.rad2deg(2*np.pi))*60
    return int(min)

# Find sec by angle
def get_sec(angle):
    sec = angle/(np.rad2deg(2*np.pi))*60
    return int(sec)

# Get farest point 
def get_true_order(order, center): 
    # Check which x,y is the farest 
    x_n = order[1][0][0]
    y_n = order[1][0][1]
    x_f = order[1][1][0]
    y_f = order[1][1][1]

    # Cal length for each (a,b) --> (c, d) to find the farest point
    length1 = np.sqrt((x_n - center[0]) ** 2 + (y_n - center[1]) ** 2) 
    length2 = np.sqrt((x_f - center[0]) ** 2 + (y_f - center[1]) ** 2) 

    if length1 > length2: 
        return ((x_f, y_f), (x_n, y_n))
    else: 
        return ((x_n, y_n), (x_f, y_f))

# Get true center --> insertion of 3 hand 
def get_true_center(points):
    return [[x, points.count(x)] for x in set(points)]
    # return (int(x), int(y))

# Get time
def findTime(img, imgO):
    try: 
        # Find the clock by the biggest contour and wrap around it with green
        contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        clock_contour = max(contours, key = cv.contourArea) 
        cv.drawContours(imgO, clock_contour, -1, (0, 255, 0), 2)

        # Find the center of clock contour and circle it with red dot
        (x, y), radius = cv.minEnclosingCircle(clock_contour)
        center = (int(x), int(y))

        # Find all the contours then sort it by dec 
        contoursL,_ = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        contour_list = [] 
        i = 0
        for cnt in contoursL:
            area = cv.contourArea(cnt)
            contour_list.append((i, area))
            i = i + 1

        # Sort contours by area 
        contour_list.sort(key = sort_by_area, reverse=True)

        # Get the three contour inside the clock contour --> mask the hand area --> to detect line later 
        handMasked = np.zeros_like(img)
        threeMasked = contour_list[2][0]
        cv.drawContours(handMasked,[contoursL[threeMasked]],0,(1),-1)

        # Convert that hand back to white for clear view 
        handMasked = (255 * handMasked).clip(0,255).astype(np.uint8)
        kernel = np.ones((1,2),np.uint8)
        handMasked = cv.erode(handMasked, kernel, 1)
        handMasked = cv.morphologyEx(handMasked, cv.MORPH_CLOSE, np.ones((1,1),np.uint8))
        erode_closing = handMasked.copy()
        # handMasked = cv.dilate(handMasked, kernel, 1)
        handMasked = skeletonize(handMasked)
        handMasked = (255 * handMasked).clip(0,255).astype(np.uint8)

        # Find the line hand on the masked img 
        lines_list = {}
        lines = cv.HoughLinesP(
                    handMasked, 
                    1, 
                    np.pi/180, 
                    threshold=35, 
                    minLineLength=50, 
                    maxLineGap=radius
                    )
        
        count_line = 0
        if lines is not None:
            count_line = len(lines_list)
            print("LINES: ", len(lines)) 
            for points in lines:
                x1, y1, x2, y2 = points[0] 
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) 
                if length > 60: 
                    # Draw hand line with red 
                    # cv.line(imgO,(x1,y1),(x2,y2),(0,0,255),2)

                    lines_list[length] = ((x1, y1), (x2, y2))

            # Find the hour, min, sec by it length: hour < min < sec 
            lines_list = sorted(lines_list.items())
            top3 = lines_list[:3]

            # The farest point of each hand --> length, ((x1, y1), (x2, y2)) --> (x2, y2)
            hour_hand = (get_true_order(top3[0], center))
            min_hand = (get_true_order(top3[1], center))
            sec_hand = (get_true_order(top3[2], center))
            
            # Get all the point on each hand 
            hour_hand_points = (np.linspace((hour_hand[0]), (hour_hand[1]), num=100, endpoint=True, dtype=int)).tolist()
            min_hand_points = (np.linspace((min_hand[0]), (min_hand[1]), num=100, endpoint=True, dtype=int)).tolist()
            sec_hand_points = (np.linspace((sec_hand[0]), (sec_hand[1]), num=100, endpoint=True, dtype=int)).tolist()
            points = hour_hand_points + min_hand_points + sec_hand_points 

            counts = []
            for i in points: 
                counts.append((i, (points.count(i))))
            counts.sort(key = sort_by_area, reverse=True)

            if counts[0][1] == 1:
                true_center = center 
            else:
                true_center = ((counts[0][0][0]), (counts[0][0][1]))

            # Draw each hand by diff color 
            # cv.line(imgO, hour_hand[0], hour_hand[1],(0, 255, 0),2) # hour --> green
            # cv.line(imgO, min_hand[0], min_hand[1],(255, 0, 0),2) # min --> blue 
            # cv.line(imgO, sec_hand[0], sec_hand[1],(0, 255, 255),2) # sec --> yellow
            
            cv.circle(imgO, hour_hand[1], 2, (0,0,0), 5)
            cv.circle(imgO, min_hand[1], 2, (230,0,255), 5)
            cv.circle(imgO, sec_hand[1], 2, (255,145,0), 5)

            cv.putText(imgO,"h", hour_hand[1], cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
            cv.putText(imgO,"m", min_hand[1], cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
            cv.putText(imgO,"s", sec_hand[1], cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv.LINE_AA)

            # Draw each hand wrapped by rectangle
            cv.rectangle(imgO, hour_hand[0], hour_hand[1],(0, 255, 0),2) # hour --> green
            cv.rectangle(imgO, min_hand[0], min_hand[1],(255, 0, 0),2) # min --> blue 
            cv.rectangle(imgO, sec_hand[0], sec_hand[1],(0, 255, 255),2) # sec --> yellow

            # Get angle 
            hour_angle = np.rad2deg(find_angle(hour_hand[1], center))
            min_angle = np.rad2deg(find_angle(min_hand[1], center))
            sec_angle = np.rad2deg(find_angle(sec_hand[1], center))
            
            if hour_angle is not None and min_angle is not None and sec_angle is not None:
                # Get time 
                hour = get_hour(hour_angle)
                min = get_min(min_angle)
                sec = get_sec(sec_angle)

                print("TIME: ", hour, min, sec)
            else: 
                hour = 0
                min = 0
                sec = 0

            h, m, s = "", "", ""
            if hour < 10: 
                h = '0' + str(hour)
            else: 
                h = str(hour)
            if min < 10: 
                m = '0' + str(min)
            else: 
                m = str(min)
            if sec < 10: 
                s = '0' + str(sec)
            else: 
                s = str(sec)
            
            time = h + ":" + m + ":" + s 
            cv.putText(imgO,time, (50,50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

            cv.circle(imgO, center, 2, (0,0,255), 2)
            cv.circle(imgO, true_center, 2, (100, 10, 79), 5)
    except: 
        print("Can't detect time")

    return erode_closing, handMasked, imgO

images_erode_closing = []
images_skeletonize = []
images_clock = []
for i in range(len(images_pre)):
    a, b, c = findTime(images_pre[i], images[i])
    images_erode_closing.append(a)
    images_skeletonize.append(b)
    images_clock.append(c)

# Save the output 
pathP = os.getcwd() + "/output/"
write_to_path_precessing(pathP + 'gray/', images_gray, 'gray')
write_to_path_precessing(pathP + 'thresh/', images_pre, 'thresh')
write_to_path_precessing(pathP + 'erode_closing/', images_erode_closing, 'erode_closing')
write_to_path_precessing(pathP + 'skeletonize/', images_skeletonize, 'skeletonize')
write_to_path(images_clock)
show(images_clock)
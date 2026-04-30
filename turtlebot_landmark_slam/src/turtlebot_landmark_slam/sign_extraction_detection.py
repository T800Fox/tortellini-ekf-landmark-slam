import cone_detection as cd
import cv2
import numpy as np

def canny_threshold_value(frame: cv2.Mat, deviation: float = 0.33) -> tuple[float, float]:
    avg_intense = np.median(frame)
    
    minVal = avg_intense * (1.0 - deviation)
    maxVal = avg_intense * (1.0 + deviation)
    
    return minVal, maxVal

def extract_horizontal_lines(edge):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    horiz = cv2.morphologyEx(edge, cv2.MORPH_OPEN, kernel, iterations=1)
    return horiz

def sign_detection_extraction_horiz(frame: cv2.Mat):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)

    # Edge detection
    canny_threshold = canny_threshold_value(gray)
    edges = cv2.Canny(gray, canny_threshold[0], canny_threshold[1])
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    horiz = extract_horizontal_lines(edges)

    contours, _ = cv2.findContours(horiz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    lines = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        aspect = w / (h + 1e-6)

        if aspect > 5:   # very wide → horizontal line
            lines.append((x, y, w, h))
            
    bands = []

    lines = sorted(lines, key=lambda b: b[1])  # sort by y

    for i in range(len(lines) - 1):
        x1, y1, w1, h1 = lines[i]
        x2, y2, w2, h2 = lines[i+1]

        vertical_gap = y2 - (y1 + h1)

        if 10 < vertical_gap < 200:  # tune this
            bands.append((lines[i], lines[i+1]))
    
    rois = []

    for top, bottom in bands:

        x1, y1, w1, h1 = top
        x2, y2, w2, h2 = bottom

        x = min(x1, x2)
        w = max(x1 + w1, x2 + w2) - x

        y = y1 + h1
        h = y2 - y

        if h > 0 and w > 0:
            roi = frame[y:y+h, x:x+w]
            rois.append(roi)
    return edges, rois

def sign_detection_extraction(frame: cv2.Mat):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)

    # Edge detection
    canny_threshold = canny_threshold_value(gray)
    edges = cv2.Canny(gray, canny_threshold[0], canny_threshold[1])
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Convert edges to BGR so we can draw colored boxes
    edges_with_boxes = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    crops = []
    image_size = frame.shape[0] * frame.shape[1]

    for cnt in contours:
        if cv2.contourArea(cnt) < 0.05 * image_size:  # filter out small contours that are unlikely to be signs
            continue

        #if not is_circle_like(cnt, 0.75):
        #    continue

        # Clean contour
        hull = cv2.convexHull(cnt)

        x, y, w, h = cv2.boundingRect(hull)

        # Draw bounding box on edge image
        cv2.rectangle(
            edges_with_boxes,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),  # green box
            2
        )

        # Crop from original image
        crop = frame[y:y+h, x:x+w]
        crops.append(crop)

    return edges_with_boxes, crops

def is_circle_like(cnt, min_circularity=0.75):
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)

    if peri == 0:
        return False

    circularity = 4 * np.pi * area / (peri * peri)

    return circularity > min_circularity

def square_contour_filter(contours: list, signs: list) -> tuple[list, list]:
    valid_squares_contours = []
    valid_signs = []
    
    for box, sign in zip(contours, signs):
        # width to height ratio condition
        w_h = ((sign.shape[0] / sign.shape[1]) > 1.4) or ((sign.shape[0] / sign.shape[1]) < 0.6)
        
        # height to width ratio condition
        h_w = ((sign.shape[1] / sign.shape[0]) > 1.4) or ((sign.shape[1] / sign.shape[0]) < 0.6)
        if (w_h and h_w):
            continue # filter out non-square shapes, as traffic signs are usually square or circular (aspect ratio close to 1)
        valid_squares_contours.append(box)
        valid_signs.append(sign)
        
    return valid_squares_contours, valid_signs

def small_contour_filter(org_img: cv2.Mat, contours: list, signs: list, minContourRatio: float) -> tuple[list, list]:
    valid_contours = []
    valid_signs = []
    
    image_size = org_img.shape[0] * org_img.shape[1]
    
    for box, sign in zip(contours, signs):
        sign_area = sign.shape[0] * sign.shape[1]
        if sign_area < minContourRatio * image_size:
            continue
        valid_contours.append(box)
        valid_signs.append(sign)
        
    return valid_contours, valid_signs

def is_box_inside_image(box, img_shape):
    h, w = img_shape[:2]

    for (x, y) in box:
        if x < 0 or x >= w or y < 0 or y >= h:
            return False
    return True

def filter_outofframe_boxes(org_img: cv2.Mat, boxes: list, warped: list) -> tuple[list, list]:
    valid_boxes = []
    valid_warped = []
    for box, warp in zip(boxes, warped):
        # Reject boxes outside image
        if not is_box_inside_image(box, org_img.shape):
            continue
        valid_boxes.append(box)
        valid_warped.append(warp)
        
    return valid_boxes, valid_warped

def color_sign_filter(contours: list, signs: list) -> tuple[list, list]:
    valid_contours = []
    valid_signs = []
    
    for box, sign in zip(contours, signs):
        # Convert to HSV color space for better color filtering
        hsv = cv2.cvtColor(sign, cv2.COLOR_BGR2HSV)

        # Define color range for blue
        minBlue = np.array([105, 70, 20])
        maxBlue = np.array([125, 255, 255])
        maskblue = cv2.inRange(hsv, minBlue, maxBlue)
        
        minWhite = np.array([5, 10, 10])
        maxWhite = np.array([30, 255, 255])
        maskwhite = cv2.inRange(hsv, minWhite, maxWhite)
        mask = cv2.bitwise_or(maskblue, maskwhite)
        # Check if the masked area is significant enough to be considered a valid sign
        if not (cv2.countNonZero(maskblue) > 0.1 * sign.shape[0] * sign.shape[1]):  # at least 10% of the area should be blue
            continue
        
        #if not (cv2.countNonZero(maskwhite) > 0.1 * sign.shape[0] * sign.shape[1]):  # at least 10% of the area should be white
         #   continue
        
        valid_contours.append(box)  
        valid_signs.append(sign)

    return valid_contours, valid_signs

def sign_pipeline(img: cv2.Mat, edge_detect: cv2.Mat, contours: list, signs: list) -> tuple[list, list]:

    contours, signs = filter_outofframe_boxes(img, contours, signs)
    contours, signs = square_contour_filter(contours, signs)
    contours, signs = small_contour_filter(img, contours, signs, 0.02)
    #contours, signs = color_sign_filter(contours, signs)
    
    # At this point we should have: 
    # - remove small contours that are unlikely to be traffic signs
    # - remove non-square contours that are unlikely to be traffic signs
    # - remove contours that are outside the image frame
    # - remove contours that do not have enough blue color (as we are looking for blue signs)

    return contours, signs
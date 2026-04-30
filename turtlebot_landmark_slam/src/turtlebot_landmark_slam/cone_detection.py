import cv2
import numpy as np
import matplotlib.pyplot as plt


class ConeDetection():
    def __init__(self, img):
        self.boxes , self.warped_images, self.mask, self.debug_img = self.cone_detect_pipeline(img)
        pass

    def morphological(self, frame: cv2.Mat) -> cv2.Mat:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
        #frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
        frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
        #frame = cv2.erode(frame, kernel, iterations=2)
        frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
        return frame
    
    def merge_boxes(self, boxes):
        merged = []
        used = [False]*len(boxes)

        for i, c1 in enumerate(boxes):
            if used[i]:
                continue

            x1,y1,w1,h1 = cv2.boundingRect(c1)
            group = [c1]

            for j,c2 in enumerate(boxes):
                if i==j or used[j]:
                    continue

                x2,y2,w2,h2 = cv2.boundingRect(c2)

                # horizontal overlap ratio
                overlap = max(
                    0,
                    min(x1+w1,x2+w2)-max(x1,x2)
                )
                overlap_ratio = overlap/min(w1,w2)

                # vertical separation
                gap = abs(y2-(y1+h1))

                similar_width = abs(w1-w2) < 0.4*max(w1,w2)

                if (
                    overlap_ratio > 0.6 and
                    gap < 120 and
                    similar_width
                ):
                    group.append(c2)
                    used[j] = True

            merged.append(
                cv2.convexHull(np.vstack(group))
            )

        return merged
        
    def cone_detection_and_extraction(self, frame: cv2.Mat, org_img: cv2.Mat, minContourRatio: float) -> list:
        contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # merge fragmented cone pieces
        contours = self.merge_boxes(contours)
        
        valid_contours = []
        warped_images = []
        
        image_size = frame.shape[0] * frame.shape[1]
        
        for cnt in contours:
            if cv2.contourArea(cnt) < minContourRatio * image_size:
                continue

            # Fix broken shapes
            hull = cv2.convexHull(cnt)

            # Fit rectangle
            rect = cv2.minAreaRect(hull)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            
            # Get width & height of rectangle
            rect_pts = self.order_points(box.astype("float32"))

            (tl, tr, br, bl) = rect_pts

            widthA = np.linalg.norm(br - bl)
            widthB = np.linalg.norm(tr - tl)
            width = int(min(widthA, widthB))

            heightA = np.linalg.norm(tr - br)
            heightB = np.linalg.norm(tl - bl)
            height = int(min(heightA, heightB))

            if width == 0 or height == 0:
                continue

            # Define destination points (straight rectangle)
            dst_pts = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype="float32")

            M = cv2.getPerspectiveTransform(rect_pts, dst_pts)

            warped = cv2.warpPerspective(org_img, M, (width, height))
            
            valid_contours.append(box)
            warped_images.append(warped)

        return valid_contours, warped_images

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]      # top-left
        rect[2] = pts[np.argmax(s)]      # bottom-right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]   # top-right
        rect[3] = pts[np.argmax(diff)]   # bottom-left

        return rect

    def red_mask(self, frame: cv2.Mat) -> cv2.Mat:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Red wraps around HSV -> two ranges
        lower1 = np.array([0, 65, 25])
        upper1 = np.array([12, 255, 255])

        lower2 = np.array([150, 65, 25])
        upper2 = np.array([179, 255, 255])

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)

        mask = cv2.bitwise_or(mask1, mask2)

        return mask

    def green_mask(self, frame: cv2.Mat) -> cv2.Mat:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower = np.array([55, 90, 90])
        upper = np.array([85, 255, 255])

        mask = cv2.inRange(hsv, lower, upper)

        return mask

    def yellow_mask(self, frame: cv2.Mat) -> cv2.Mat:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower = np.array([22, 120, 120])
        upper = np.array([35, 255, 255])

        mask = cv2.inRange(hsv, lower, upper)

        return mask

    def blue_mask(self, frame: cv2.Mat) -> cv2.Mat:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower = np.array([90, 155, 75])
        upper = np.array([100, 255, 255])

        mask = cv2.inRange(hsv, lower, upper)

        return mask

    def remove_heavily_overlapping_detections(
        self,
        boxes,
        warps,
        overlap_thresh=0.9
    ):
        """
        Removes smaller overlapping detections while keeping
        boxes and warped images aligned.
        """

        if len(boxes) <= 1:
            return boxes, warps

        keep = [True]*len(boxes)

        rects=[]
        areas=[]

        for box in boxes:
            x,y,w,h = cv2.boundingRect(box)
            rects.append((x,y,w,h))
            areas.append(w*h)

        for i in range(len(boxes)):
            if not keep[i]:
                continue

            x1,y1,w1,h1 = rects[i]

            for j in range(i+1,len(boxes)):
                if not keep[j]:
                    continue

                x2,y2,w2,h2 = rects[j]

                xi1=max(x1,x2)
                yi1=max(y1,y2)
                xi2=min(x1+w1,x2+w2)
                yi2=min(y1+h1,y2+h2)

                inter = max(0,xi2-xi1)*max(0,yi2-yi1)

                if inter == 0:
                    continue

                overlap = inter / min(
                    areas[i],
                    areas[j]
                )

                if overlap >= overlap_thresh:

                    if areas[i] < areas[j]:
                        keep[i]=False
                        break
                    else:
                        keep[j]=False

        filtered_boxes = []
        filtered_warps = []

        for i in range(len(boxes)):
            if keep[i]:
                filtered_boxes.append(
                    boxes[i]
                )
                filtered_warps.append(
                    warps[i]
                )

        return filtered_boxes, filtered_warps

    def cone_detect_pipeline(
            self,
            img: cv2.Mat,
            min_contour_ratio: float = 0.01,
            show_debug: bool = False,
            window_name: str = "cone_detection"
        ):
        """
        Full cone detection pipeline:
        - HSV red segmentation
        - Morphological cleanup
        - Contour extraction / merging
        - Bounding box generation
        - Perspective warping of cones

        Returns:
            boxes: detected cone boxes
            warped_images: rectified cone crops
            mask: final binary mask (useful for debugging)
            debug_img: image with contours drawn
        """

        # ---- HSV segmentation ----
        r_mask = self.red_mask(img)
        g_mask = self.green_mask(img)
        y_mask = self.yellow_mask(img)
        b_mask = self.blue_mask(img)
        
        # Morphological cleanup
        r_mask = self.morphological(r_mask)
        g_mask = self.morphological(g_mask)
        y_mask = self.morphological(y_mask)
        b_mask = self.morphological(b_mask)
        
        # Optional combined mask
        mask = r_mask | g_mask | y_mask | b_mask

        # ---- Cone extraction ----
        r_boxes, r_warped_images = self.cone_detection_and_extraction(r_mask, img, min_contour_ratio)
        
        g_boxes, g_warped_images = self.cone_detection_and_extraction(g_mask, img, min_contour_ratio)
        
        y_boxes, y_warped_images = self.cone_detection_and_extraction(y_mask, img, min_contour_ratio)
        
        b_boxes, b_warped_images = self.cone_detection_and_extraction(b_mask, img, min_contour_ratio)
        
        
        # Remove heavily overlapping boxes (e.g. from reflections or multiple contours on same cone)
        r_boxes, r_warped_images = self.remove_heavily_overlapping_detections(r_boxes, r_warped_images, 0.9)

        g_boxes, g_warped_images = self.remove_heavily_overlapping_detections(g_boxes, g_warped_images, 0.9)

        y_boxes, y_warped_images = self.remove_heavily_overlapping_detections(y_boxes, y_warped_images, 0.9)

        b_boxes, b_warped_images = self.remove_heavily_overlapping_detections(b_boxes, b_warped_images, 0.9)
        
        boxes = []

        for box in r_boxes:
            boxes.append((box, "red"))

        for box in g_boxes:
            boxes.append((box, "green"))

        for box in y_boxes:
            boxes.append((box, "yellow"))

        for box in b_boxes:
            boxes.append((box, "blue"))

        '''
        # Example of how to use the color labels for further filtering or processing
        [(box1,"red"), (box2,"red"), (box3,"blue")]
        '''
        
        warped_images = []
        for warp in r_warped_images:
            warped_images.append((warp, "red"))
            
        for warp in g_warped_images:
            warped_images.append((warp, "green"))
            
        for warp in y_warped_images:
            warped_images.append((warp, "yellow"))
            
        for warp in b_warped_images:
            warped_images.append((warp, "blue"))
    
        # Debug visualization
        debug_img = img.copy()

        # OpenCV uses BGR colors
        draw_colors = {
            "red": (0,0,255),
            "green": (0,255,0),
            "yellow": (0,255,255),
            "blue": (255,0,0)
        }

        for box, label in boxes:
            cv2.drawContours(
                debug_img,
                [box],
                -1,
                draw_colors[label],
                2
            )

        if show_debug:
            cv2.imshow(f"{window_name} - Original", img)
            cv2.imshow(f"{window_name} - Mask", mask)
            cv2.imshow(f"{window_name} - Detected", debug_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return boxes, warped_images, mask, debug_img
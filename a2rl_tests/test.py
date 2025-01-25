# Disable YOLO verbose output
import os
os.environ['YOLO_VERBOSE'] = 'False'

import cv2
import glob
import numpy as np
from pypopsift import popsift
from ultralytics import YOLO

config = {
    'sift_peak_threshold': 0.1,
    'sift_edge_threshold': 10.0,
    'feature_min_frames': 500, # keypoints
    'feature_use_adaptive_suppression': False,
    'feature_process_size': 900 # resize image to this size before processing
}

def resized_image(image, config):
    """Resize image to feature_process_size."""
    max_size = config['feature_process_size']
    h, w, _ = image.shape
    size = max(w, h)
    if 0 < max_size < size:
        dsize = w * max_size // size, h * max_size // size
        return cv2.resize(image, dsize=dsize, interpolation=cv2.INTER_AREA)
    else:
        return image

def extract_features(image, config):
    image = resized_image(image, config)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    points, desc = popsift(image.astype(np.uint8),  # values between 0, 1
                                peak_threshold=config['sift_peak_threshold'],
                                edge_threshold=config['sift_edge_threshold'],
                                target_num_features=config['feature_min_frames'])
    return points, desc, image

def match_features(matcher, desc1, desc2):
    GOOD_MATCH_PERCENT = 0.15
    matches = matcher.match(desc1, desc2)
    matches2=sorted(matches,key=lambda x: x.distance, reverse=False)
    numGoodMatches = int(len(matches2) * GOOD_MATCH_PERCENT)
    matches2 = matches2[:numGoodMatches]  # keep only the best matches
    return matches2

def convert_points(points):
    return [cv2.KeyPoint(x, y, s) for x, y, s, a in points]

def draw_matches(image1, image2, points1, points2, matches):
    image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    points1 = convert_points(points1)
    points2 = convert_points(points2)
    draw = cv2.drawMatches(image1, points1, image2, points2, matches, None)
    return draw

def preprocess_image(model, image):
    # Remove everything outside the bounding box (make it black)
    results = model(image, device="cuda:0")
    def get_area(box):
        x1, y1, x2, y2 = box.xyxy.cpu().numpy().flatten()
        return (x2-x1) * (y2-y1)
    for result in results:
        max_box = sorted(result.boxes, key=lambda x: get_area(x) if x.conf>0.5 else x[-1], reverse=True)[0]
        x1, y1, x2, y2 = max_box.xyxy.cpu().numpy().flatten()
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[int(y1):int(y2), int(x1):int(x2)] = 255 if len(image.shape) == 2 else (255, 255, 255)
    image = cv2.bitwise_and(image, mask)
    return image

def match_images(image1, image2, matcher, model, config):
    image1 = cv2.imread(image1)
    image2 = cv2.imread(image2)
    image2 = preprocess_image(model, image2)
    points1, desc1, image1 = extract_features(image1, config)
    points2, desc2, image2 = extract_features(image2, config)
    matches = match_features(matcher, desc1, desc2)
    draw = draw_matches(image1, image2, points1, points2, matches)
    cv2.imshow("Matches", draw)
    cv2.waitKey(1) & 0xFF

def get_images(folder):
    images = []
    for filename in glob.glob(folder + '/*.jpg'):
        images.append(filename)
    return images

if __name__ == "__main__":
    # match_images("img1.png", "img2.png", config)
    model = YOLO("epoch18.pt")
    #-----------------Test-----------------
    # image = cv2.imread("test_footage/302.jpg")
    # image = preprocess_image(model, image)
    # cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    # imS = cv2.resizeWindow("Image", 1920, 1080)
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #--------------------------------------
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    images = get_images("test_footage2")
    for i in images:
        try:
            match_images("gate.png", i, bf, model, config)
        except KeyboardInterrupt:
            break
        except:
            pass
    cv2.destroyAllWindows()



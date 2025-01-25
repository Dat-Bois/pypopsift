import cv2
import glob
import numpy as np
from pypopsift import popsift

config = {
    'sift_peak_threshold': 0.1,
    'sift_edge_threshold': 10.0,
    'feature_min_frames': 1000,
    'feature_use_adaptive_suppression': False,
    'feature_process_size': 900
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

def extract_features(filename, config):
    image = cv2.imread(filename)

    if image is None:
        raise IOError("Unable to load image {}".format(filename))

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

def match_images(image1, image2, matcher, config):
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
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    # match_images("img1.png", "img2.png", config)
    images = get_images("test_footage")
    for i in images:
        try:
            match_images("gate.png", i, bf, config)
        except KeyboardInterrupt:
            break
        except:
            pass
    cv2.destroyAllWindows()



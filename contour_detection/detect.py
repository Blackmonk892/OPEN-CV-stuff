"""
arrow_detector.py
Detect arrows in an image using OpenCV (contour + geometry heuristic method).

Requires:
    pip install opencv-python numpy

Usage:
    python arrow_detector.py input.jpg --out out.png

The script:
- preprocesses (grayscale/blur/threshold or Canny or HSV color mask)
- finds contours
- simplifies contours with approxPolyDP
- finds a candidate tip vertex (smallest interior angle)
- validates candidate using area, solidity, aspect ratio and centroid-distance checks
- optionally uses convexity defects as an extra filter
- draws contour, tip, centroid, and orientation on an output image
- returns a list of detected arrows with debug info
"""

import cv2
import numpy as np
import math
import argparse
from typing import List, Tuple, Dict, Optional

# ---------- Utility math helpers ----------

def angle_between(a: Tuple[int,int], b: Tuple[int,int], c: Tuple[int,int]) -> float:
    """
    Return interior angle in degrees at point b formed by points a-b-c.
    a, b, c are (x,y).
    """
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    c = np.array(c, dtype=np.float64)
    v1 = a - b
    v2 = c - b
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-9
    cosang = np.dot(v1, v2) / denom
    cosang = float(np.clip(cosang, -1.0, 1.0))
    ang_rad = math.acos(cosang)
    return math.degrees(ang_rad)

def contour_centroid(contour: np.ndarray) -> Tuple[float,float]:
    """
    Compute centroid (cx, cy) of a contour using image moments.
    """
    M = cv2.moments(contour)
    if M['m00'] != 0:
        return (M['m10']/M['m00'], M['m01']/M['m00'])
    else:
        # fallback: mean of points
        pts = contour.reshape(-1, 2)
        return (float(np.mean(pts[:,0])), float(np.mean(pts[:,1])))

# ---------- Preprocessing helpers ----------

def mask_color_hsv(img: np.ndarray, lower_hsv: Tuple[int,int,int], upper_hsv: Tuple[int,int,int]) -> np.ndarray:
    """
    Return a binary mask isolating pixels within the HSV range.
    Accepts scalar ranges (no wrap-around). If you need red (wrap), call twice and OR masks.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array(lower_hsv, dtype=np.uint8)
    upper = np.array(upper_hsv, dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)

def preprocess_image(img: np.ndarray,
                     method: str = 'adaptive',   # 'adaptive', 'otsu', or 'canny' or 'binary'
                     blur_ksize: int = 5,
                     adaptive_blocksize: int = 15,
                     adaptive_C: int = 9,
                     canny_thresh1: int = 50,
                     canny_thresh2: int = 150,
                     use_color_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Produce a binary image ready for contour extraction.
    - method:
        'adaptive' -> adaptiveThreshold on blurred grayscale (good for varying light)
        'otsu'     -> global Otsu threshold
        'canny'    -> Canny edges (useful for line-drawn arrows)
        'binary'   -> simple global threshold (for clean images)
    - use_color_mask: optional binary mask (same size as input) to restrict to a color region
    Returns: binary image (uint8) where shapes are white (255) on black (0)
    """
    assert blur_ksize % 2 == 1, "blur_ksize must be odd"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    if method == 'adaptive':
        th = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV,
                                   blockSize=adaptive_blocksize,
                                   C=adaptive_C)
        binary = th
    elif method == 'otsu':
        _, th = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary = th
    elif method == 'binary':
        # simple threshold at a chosen value (useful if you know contrast)
        _, th = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
        binary = th
    elif method == 'canny':
        edges = cv2.Canny(blurred, canny_thresh1, canny_thresh2)
        # close gaps so contours are continuous
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        binary = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    else:
        raise ValueError("method must be 'adaptive','otsu','binary', or 'canny'")

    # If a color mask is provided (e.g., from HSV), combine it: only keep binary pixels where mask is true
    if use_color_mask is not None:
        # Ensure mask is single channel binary 0/255
        mask = use_color_mask.copy()
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = (mask > 0).astype(np.uint8) * 255
        binary = cv2.bitwise_and(binary, mask)

    # final morphological close to fill small holes and make contours smoother
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    # optionally we could do opening to remove small specks, but closing first is safer for arrow heads
    return binary

# ---------- Contour & geometry logic ----------

def find_contours(binary_img: np.ndarray, min_area: int = 500) -> List[np.ndarray]:
    """
    Find external contours and filter by area. Returns list of contours (numpy arrays).
    """
    # Note: OpenCV findContours returns either (contours, hierarchy) or (image, contours, hierarchy)
    res = cv2.findContours(binary_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = res[0] if len(res) == 2 else res[1]
    filtered = [c for c in contours if cv2.contourArea(c) >= min_area]
    # sort by area descending (bigger shapes first)
    filtered.sort(key=cv2.contourArea, reverse=True)
    return filtered

def approx_vertices(contour: np.ndarray, epsilon_factor: float = 0.01) -> np.ndarray:
    """
    Return approxPolyDP vertices for a contour. epsilon_factor is proportion of contour perimeter.
    """
    peri = cv2.arcLength(contour, True)
    eps = epsilon_factor * peri
    approx = cv2.approxPolyDP(contour, eps, True)
    return approx.reshape(-1, 2)

def convexity_defects_info(contour: np.ndarray) -> Dict:
    """
    Compute convex hull (indices) and convexity defects for a contour.
    Returns dict with keys: 'defects' (list), 'num_large_defects' (int), 'max_depth' (float)
    Each defect entry is (start_idx, end_idx, far_idx, depth)
    depth is in pixels (distance * 256 in some OpenCV versions, but here we trust returned depth)
    """
    res = {'defects': [], 'num_large_defects': 0, 'max_depth': 0.0}
    if contour.shape[0] < 4:
        return res
    hull_idx = cv2.convexHull(contour, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 3:
        return res
    defects = cv2.convexityDefects(contour, hull_idx)
    if defects is None:
        return res
    di = []
    for i in range(defects.shape[0]):
        s, e, f, depth = defects[i, 0]
        di.append((int(s), int(e), int(f), float(depth)))
    res['defects'] = di
    if len(di) > 0:
        depths = np.array([d[-1] for d in di], dtype=np.float64)
        res['max_depth'] = float(depths.max())
    # a "large" defect threshold could be used relative to contour size when evaluating arrow tails
    return res

def find_tip_point(contour: np.ndarray,
                   epsilon_factor: float = 0.01,
                   angle_threshold: float = 60.0,
                   dist_ratio_threshold: float = 0.45) -> Tuple[Optional[Tuple[int,int]], Optional[float]]:
    """
    Heuristic to find arrow tip (the sharpest vertex).
    - Use approxPolyDP to simplify contour.
    - For each vertex compute interior angle; the smallest angle is candidate tip.
    - Ensure candidate is far enough from centroid to avoid small dents.
    Returns (tip_point (x,y) or None, min_angle or None).
    """
    pts = approx_vertices(contour, epsilon_factor=epsilon_factor)
    if pts.shape[0] < 3:
        return None, None

    cx, cy = contour_centroid(contour)
    centroid = np.array([cx, cy])
    dists = np.linalg.norm(pts - centroid, axis=1)
    max_dist = float(dists.max()) if len(dists) > 0 else 1.0

    min_angle = 180.0
    min_idx = -1
    n = pts.shape[0]
    for i in range(n):
        prev = tuple(pts[(i-1) % n])
        cur = tuple(pts[i])
        nxt = tuple(pts[(i+1) % n])
        ang = angle_between(prev, cur, nxt)
        if ang < min_angle:
            min_angle = ang
            min_idx = i

    if min_idx == -1:
        return None, None

    tip_pt = tuple(int(v) for v in pts[min_idx])
    tip_dist = float(np.linalg.norm(pts[min_idx] - centroid))

    # Conditions: angle small enough AND tip far enough from centroid relative to shape size
    if min_angle < angle_threshold and tip_dist > dist_ratio_threshold * max_dist:
        return tip_pt, min_angle
    else:
        return None, min_angle

def is_arrow_contour(contour: np.ndarray,
                     min_area: float = 500.0,
                     angle_threshold: float = 60.0,
                     dist_ratio_threshold: float = 0.45,
                     solidity_max: float = 0.97,
                     aspect_ratio_min: float = 1.05,
                     require_convex_defect: bool = False,
                     convex_defect_depth_ratio: float = 0.02) -> Tuple[bool, Dict]:
    """
    Heuristic composite check to decide whether a contour likely represents an arrow.
    Returns (is_arrow (bool), debug_info (dict)).
    debug_info holds metrics used to make the decision so you can tune thresholds.
    """
    debug = {}
    area = float(cv2.contourArea(contour))
    debug['area'] = area
    if area < min_area:
        debug['reason'] = 'area_too_small'
        return False, debug

    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull)) if cv2.contourArea(hull) > 0 else 1.0
    solidity = area / hull_area
    debug['solidity'] = solidity

    x,y,w,h = cv2.boundingRect(contour)
    aspect_ratio = float(max(w,h)) / (min(w,h) + 1e-8)
    debug['aspect_ratio'] = aspect_ratio
    debug['bounding_box'] = (x,y,w,h)

    tip, min_angle = find_tip_point(contour,
                                    epsilon_factor=0.01,
                                    angle_threshold=angle_threshold,
                                    dist_ratio_threshold=dist_ratio_threshold)
    debug['tip'] = tip
    debug['min_angle'] = min_angle

    # optionally compute convexity defects info
    defects_info = convexity_defects_info(contour)
    debug['convexity'] = defects_info

    # Basic heuristics:
    # 1) tip must exist (sharp enough and far enough)
    if tip is None:
        debug['reason'] = 'no_tip'
        return False, debug

    # 2) shape should be somewhat elongated (not a nearly-square blob)
    if aspect_ratio < aspect_ratio_min:
        debug['reason'] = 'not_elongated'
        return False, debug

    # 3) solidity should be less than a threshold (arrow has concavity at tail)
    if solidity >= solidity_max:
        debug['reason'] = 'too_convex'
        return False, debug

    # 4) optional: expect at least one significant convexity defect (tail region)
    if require_convex_defect:
        diag = math.hypot(w, h)
        large_defects = [d for d in defects_info['defects'] if d[3] > convex_defect_depth_ratio * diag]
        debug['large_defects_count'] = len(large_defects)
        if len(large_defects) == 0:
            debug['reason'] = 'no_significant_convexity_defects'
            return False, debug

    # if all checks passed
    debug['reason'] = 'passed'
    return True, debug

# ---------- Main detection function ----------

def detect_arrows(img: np.ndarray,
                  preprocess_params: Dict = None,
                  contour_params: Dict = None,
                  draw_params: Dict = None) -> Tuple[np.ndarray, List[Dict]]:
    """
    Main function. Returns (annotated_image, list_of_arrow_info)
    Each arrow_info is a dict with keys:
      - contour: numpy array
      - tip: (x,y)
      - min_angle: degrees
      - centroid: (cx, cy)
      - bbox: (x,y,w,h)
      - area: float
      - orientation_deg: float (0..360 with 0 -> pointing right, measured CCW)
      - debug: debug dict from is_arrow_contour
    """
    # default parameters
    if preprocess_params is None:
        preprocess_params = {'method': 'adaptive', 'blur_ksize': 5, 'adaptive_blocksize': 15, 'adaptive_C': 9}
    if contour_params is None:
        contour_params = {'min_area': 800, 'angle_threshold': 60.0, 'dist_ratio_threshold': 0.45,
                          'solidity_max': 0.97, 'aspect_ratio_min': 1.05, 'require_convex_defect': False}
    if draw_params is None:
        draw_params = {'contour_color': (0,255,0), 'tip_color': (0,0,255), 'centroid_color': (255,0,0),
                       'thickness': 2, 'tip_radius': 6}

    # Optional color mask passed to preprocess_params: 'color_mask' key (numpy array)
    color_mask = preprocess_params.get('color_mask', None)

    binary = preprocess_image(img,
                              method=preprocess_params.get('method', 'adaptive'),
                              blur_ksize=preprocess_params.get('blur_ksize', 5),
                              adaptive_blocksize=preprocess_params.get('adaptive_blocksize', 15),
                              adaptive_C=preprocess_params.get('adaptive_C', 9),
                              canny_thresh1=preprocess_params.get('canny_thresh1', 50),
                              canny_thresh2=preprocess_params.get('canny_thresh2', 150),
                              use_color_mask=color_mask)

    contours = find_contours(binary, min_area=contour_params.get('min_area', 800))

    out = img.copy()
    detected = []

    for c in contours:
        is_arrow, debug = is_arrow_contour(c,
                                           min_area=contour_params.get('min_area', 800),
                                           angle_threshold=contour_params.get('angle_threshold', 60.0),
                                           dist_ratio_threshold=contour_params.get('dist_ratio_threshold', 0.45),
                                           solidity_max=contour_params.get('solidity_max', 0.97),
                                           aspect_ratio_min=contour_params.get('aspect_ratio_min', 1.05),
                                           require_convex_defect=contour_params.get('require_convex_defect', False),
                                           convex_defect_depth_ratio=contour_params.get('convex_defect_depth_ratio', 0.02))
        debug['contour_area'] = float(cv2.contourArea(c))
        if is_arrow:
            tip = debug.get('tip')  # tip from debug (tuple) or None
            # centroid
            cx, cy = contour_centroid(c)
            # orientation vector (from centroid to tip)
            if tip is not None:
                tx, ty = tip
                vx = tx - cx
                vy = cy - ty   # note: invert y to get standard math coordinates (y up)
                # angle in degrees, 0 = right, positive CCW
                ang = math.degrees(math.atan2(vy, vx))
                ang = (ang + 360.0) % 360.0
            else:
                ang = None

            # draw contour
            cv2.drawContours(out, [c], -1, draw_params['contour_color'], draw_params['thickness'])
            # draw tip
            if tip is not None:
                cv2.circle(out, (int(tip[0]), int(tip[1])), draw_params['tip_radius'], draw_params['tip_color'], -1)
                # draw a small line showing orientation from tip towards centroid
                cv2.line(out, (int(tip[0]), int(tip[1])), (int(cx), int(cy)), draw_params['contour_color'], 1)
            # draw centroid
            cv2.circle(out, (int(cx), int(cy)), 4, draw_params['centroid_color'], -1)

            x,y,w,h = debug.get('bounding_box', (0,0,0,0))
            info = {
                'contour': c,
                'tip': tip,
                'min_angle': debug.get('min_angle'),
                'centroid': (cx, cy),
                'bbox': (x,y,w,h),
                'area': float(cv2.contourArea(c)),
                'orientation_deg': ang,
                'debug': debug
            }
            detected.append(info)
        # else: skip non-arrows

    return out, detected

# ---------- CLI and example ----------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('image', help='path to input image')
    p.add_argument('--out', default='arrows_out.png', help='output annotated image path')
    p.add_argument('--min_area', type=int, default=800, help='minimum contour area to consider')
    p.add_argument('--use_canny', action='store_true', help='use Canny preprocessing instead of adaptive threshold')
    p.add_argument('--visualize', action='store_true', help='show result window (may not work on headless servers)')
    p.add_argument('--hsv_mask', nargs=6, type=int,
                   metavar=('H1','S1','V1','H2','S2','V2'),
                   help='optional HSV mask range (H1 S1 V1 H2 S2 V2). Use to filter by color before shape detection.')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read image: {args.image}")

    preprocess_params = {'method': 'canny' if args.use_canny else 'adaptive', 'blur_ksize': 5}
    if args.hsv_mask:
        h1,s1,v1,h2,s2,v2 = args.hsv_mask
        mask = mask_color_hsv(img, (h1,s1,v1), (h2,s2,v2))
        preprocess_params['color_mask'] = mask

    contour_params = {'min_area': args.min_area, 'angle_threshold': 60.0, 'dist_ratio_threshold': 0.45,
                      'solidity_max': 0.97, 'aspect_ratio_min': 1.05, 'require_convex_defect': False}

    annotated, found = detect_arrows(img, preprocess_params=preprocess_params, contour_params=contour_params)

    cv2.imwrite(args.out, annotated)
    print(f"Saved annotated image to {args.out}")
    print(f"Detected {len(found)} arrow(s).")
    for i,info in enumerate(found):
        print(f" Arrow {i+1}: tip={info['tip']}, angle={info['min_angle']:.1f}°, orientation={info['orientation_deg']}, area={info['area']:.1f}")
    if args.visualize:
        cv2.imshow('Annotated', annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

import cv2
import numpy as np


# ============================================================
# Helper: resize so manual loops are fast enough to be "live"
# ============================================================
def resize(image, width=400):
    h, w = image.shape[:2]
    scale = width / w
    return cv2.resize(image, (width, int(h * scale)))


# ============================================================
# 1. Sobel Edge Detection
# ------------------------------------------------------------
# Use two 3x3 kernels: one for horizontal edges (Sx),
# one for vertical edges (Sy). Convolve the image with each,
# then combine using the gradient magnitude formula:
#     G = sqrt(Gx^2 + Gy^2)
# ============================================================
def sobel(gray):
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float64)

    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=np.float64)

    Gx = cv2.filter2D(gray.astype(np.float64), -1, sobel_x)
    Gy = cv2.filter2D(gray.astype(np.float64), -1, sobel_y)

    G = np.sqrt(Gx ** 2 + Gy ** 2)

    G = (G / G.max()) * 255
    return G.astype(np.uint8)


# ============================================================
# 2. Prewitt Edge Detection
# ------------------------------------------------------------
# Same idea as Sobel but with simpler kernels (all weights = 1).
# ============================================================
def prewitt(gray):
    prewitt_x = np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]], dtype=np.float64)

    prewitt_y = np.array([[-1, -1, -1],
                          [ 0,  0,  0],
                          [ 1,  1,  1]], dtype=np.float64)

    Gx = cv2.filter2D(gray.astype(np.float64), -1, prewitt_x)
    Gy = cv2.filter2D(gray.astype(np.float64), -1, prewitt_y)

    G = np.sqrt(Gx ** 2 + Gy ** 2)

    G = (G / G.max()) * 255
    return G.astype(np.uint8)


# ============================================================
# 3. Canny Edge Detection
# ------------------------------------------------------------
# Steps performed by cv2.Canny internally:
#   1. Gaussian blur to reduce noise.
#   2. Sobel gradients.
#   3. Non-maximum suppression (thin the edges).
#   4. Double thresholding (low & high).
#   5. Hysteresis (connect strong edges to weak ones).
# ============================================================
def canny(gray):
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    low_threshold = 100
    high_threshold = 200
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    return edges


# ============================================================
# 4. Object Detection (Vehicles)
# ------------------------------------------------------------
#   1. Convert to grayscale.
#   2. Blur to remove small noise.
#   3. Get edges with Canny.
#   4. Dilate edges so broken lines join.
#   5. Find external contours.
#   6. Keep only contours that look like vehicles
#      (not too small, not too big, sensible width/height ratio).
#   7. Draw bounding boxes around them.
# ============================================================
def objects(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 80, 180)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    output = bgr.copy()
    h, w = bgr.shape[:2]
    count = 0

    for c in contours:
        area = cv2.contourArea(c)
        if h * w * 0.005 < area < h * w * 0.35:
            x, y, bw, bh = cv2.boundingRect(c)
            ratio = bw / bh
            if 0.3 < ratio < 4.0:
                cv2.rectangle(output, (x, y), (x + bw, y + bh),
                              (0, 255, 0), 2)
                count += 1

    cv2.putText(output, f"Objects: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return output


# ============================================================
# 5. Road Segmentation
# ------------------------------------------------------------
#   1. Convert BGR -> HSV (better for color filtering).
#   2. Build a mask of pixels that fall inside an
#      "asphalt-grey" HSV range.
#   3. Clean the mask with morphological OPEN then CLOSE.
#   4. Blend a red color on top of the mask area.
# ============================================================
def road(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    lower = np.array([0,   0,  40])
    upper = np.array([180, 60, 200])
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    output = bgr.copy()
    red = np.array([0, 0, 255], dtype=np.float64)
    output[mask > 0] = (0.5 * output[mask > 0] + 0.5 * red).astype(np.uint8)
    return output


# ============================================================
# 6. Harris Corner Detection  (from scratch)
# ------------------------------------------------------------
#   1. Compute gradients Ix, Iy with Sobel.
#   2. Compute Ix^2, Iy^2, IxIy.
#   3. Sum each over a small window.
#   4. Harris response:  R = det(M) - k * trace(M)^2
#      where det(M)  = Sx2 * Sy2 - Sxy^2
#            trace(M) = Sx2 + Sy2
#   5. Normalize R and threshold to mark corners.
# ============================================================
def harris(bgr, k=0.04, window_size=3, threshold=0.01):
    bgr_small = resize(bgr, 300)
    image = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2GRAY).astype(np.float64)
    height, width = image.shape

    # ---- 1. Gradients (Sobel) -----------------------------
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float64)

    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=np.float64)

    padded = np.pad(image, 1, mode='constant')
    Ix = np.zeros_like(image)
    Iy = np.zeros_like(image)
    for i in range(height):
        for j in range(width):
            Ix[i, j] = np.sum(padded[i:i+3, j:j+3] * sobel_x)
            Iy[i, j] = np.sum(padded[i:i+3, j:j+3] * sobel_y)

    # ---- 2. Products of derivatives -----------------------
    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    Ixy = Ix * Iy

    # ---- 3. Window summation ------------------------------
    offset = window_size // 2
    R = np.zeros_like(image)
    for i in range(offset, height - offset):
        for j in range(offset, width - offset):
            Sx2 = np.sum(Ix2[i-offset:i+offset+1, j-offset:j+offset+1])
            Sy2 = np.sum(Iy2[i-offset:i+offset+1, j-offset:j+offset+1])
            Sxy = np.sum(Ixy[i-offset:i+offset+1, j-offset:j+offset+1])

            # ---- 4. Harris response ------------------------
            det = (Sx2 * Sy2) - (Sxy ** 2)
            trace = Sx2 + Sy2
            R[i, j] = det - k * (trace ** 2)

    # ---- 5. Normalize & threshold -------------------------
    R_norm = R / R.max()
    output = bgr_small.copy()
    output[R_norm > threshold] = (0, 0, 255)
    return output


# ============================================================
# 7. Template Matching
# ------------------------------------------------------------
#   - Slide a small template across the larger image.
#   - At each location compute Normalized Cross-Correlation.
#   - Highest correlation = best match location.
#   - Draw a rectangle around the match.
# ============================================================
def template_match(bgr, tpl=None):
    if tpl is None:
        h, w = bgr.shape[:2]
        tpl = bgr[h//2 - h//12:h//2 + h//12,
                  w//2 - w//12:w//2 + w//12]

    gray_image    = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(gray_image, gray_template,
                               cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(result)

    th, tw = gray_template.shape
    output = bgr.copy()
    cv2.rectangle(output, loc, (loc[0] + tw, loc[1] + th),
                  (255, 0, 0), 3)
    cv2.putText(output, f"Match: {score:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    return output, tpl

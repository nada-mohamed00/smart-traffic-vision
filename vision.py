import cv2
import numpy as np


def sobel(gray):
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return cv2.convertScaleAbs(cv2.magnitude(gx, gy))


def prewitt(gray):
    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32)
    ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], np.float32)
    gx = cv2.filter2D(gray, cv2.CV_64F, kx)
    gy = cv2.filter2D(gray, cv2.CV_64F, ky)
    return cv2.convertScaleAbs(cv2.magnitude(gx, gy))


def canny(gray):
    return cv2.Canny(gray, 100, 200)


def objects(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 80, 180)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = bgr.copy()
    h, w = bgr.shape[:2]
    n = 0
    for c in cnts:
        a = cv2.contourArea(c)
        if h * w * 0.005 < a < h * w * 0.35:
            x, y, bw, bh = cv2.boundingRect(c)
            if 0.3 < bw / bh < 4.0:
                cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                n += 1
    cv2.putText(out, f"Objects: {n}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return out


def road(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 40]), np.array([180, 60, 200]))
    k = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    out = bgr.copy()
    out[mask > 0] = (0.5 * out[mask > 0] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
    return out


def harris(bgr):
    gray = np.float32(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    r = cv2.cornerHarris(gray, 2, 3, 0.04)
    r = cv2.dilate(r, None)
    out = bgr.copy()
    out[r > 0.01 * r.max()] = (0, 0, 255)
    return out


def template_match(bgr, tpl=None):
    if tpl is None:
        h, w = bgr.shape[:2]
        tpl = bgr[h // 2 - h // 12:h // 2 + h // 12, w // 2 - w // 12:w // 2 + w // 12]
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    t = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    th, tw = t.shape
    out = bgr.copy()
    cv2.rectangle(out, loc, (loc[0] + tw, loc[1] + th), (255, 0, 0), 3)
    cv2.putText(out, f"Match: {score:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    return out, tpl

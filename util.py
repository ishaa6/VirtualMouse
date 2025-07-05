import numpy as np

def get_angle(a, b, c):
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    return angle

def get_distance(landmark_lst):
    if (len(landmark_lst)<2):
        return
    (x1, y1), (x2, y2) = landmark_lst[0], landmark_lst[1]
    L = np.hypot(x2-x1, y2-y1)
    return np.interp(L, [0,1], [0,1000])
import cv2
import numpy as np

print('MORPH_HITMISS', cv2.MORPH_HITMISS)

binimg = np.zeros((10, 10), dtype=np.uint8)
binimg[2:6, 2:6] = 1

k1 = np.array([
    [-1, -1, -1, -1, -1, -1],
    [-1,  1,  1,  1,  1, -1],
    [-1,  1,  1,  1,  1, -1],
    [-1,  1,  1,  1,  1, -1],
    [-1,  1,  1,  1,  1, -1],
    [-1, -1, -1, -1, -1, -1]
], dtype=np.int8)

k2 = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0]
], dtype=np.int8)

res1 = cv2.morphologyEx(binimg, cv2.MORPH_HITMISS, k1, anchor=(1, 1))
res2 = cv2.morphologyEx(binimg, cv2.MORPH_HITMISS, k2, anchor=(1, 1))

print('sum1', np.sum(res1))
print('sum2', np.sum(res2))
print('res1:\n', res1)
print('res2:\n', res2)

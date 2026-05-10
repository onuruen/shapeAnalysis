import cv2
import numpy as np

image = cv2.imread('images/Shapes1.bmp')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold - foreground=1, background=0
_, binary = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY)

# HMT kernel
hmt_kernel_correct = np.array([
    [-1, -1, -1, -1, -1, -1],
    [-1,  1,  1,  1,  1, -1],
    [-1,  1,  1,  1,  1, -1],
    [-1,  1,  1,  1,  1, -1],
    [-1,  1,  1,  1,  1, -1],
    [-1, -1, -1, -1, -1, -1]
], dtype=np.int8)

hit_miss = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_HITMISS, hmt_kernel_correct, anchor=(-1, -1))

# Dilation with anchor (3,3)
dilation_kernel = np.ones((4, 4), dtype=np.uint8)
dilated = cv2.dilate(hit_miss.astype(np.uint8), dilation_kernel, anchor=(3, 3))

# Debug output
print(f"Binary range: {binary.min()}-{binary.max()}")
print(f"HMT range: {hit_miss.min()}-{hit_miss.max()}")
print(f"Dilated range: {dilated.min()}-{dilated.max()}")
print(f"Binary sum: {binary.sum()}, Dilated sum: {dilated.sum()}")

overlap = np.sum((dilated == 1) & (binary == 1))
dilated_outside = np.sum((dilated == 1) & (binary == 0))
print(f"Overlap: {overlap}, Dilated outside binary: {dilated_outside}")

# Check a sample square location
locs = np.argwhere(hit_miss == 1)
if len(locs) > 0:
    sample_r, sample_c = locs[0]
    print(f"\nSample square at ({sample_r}, {sample_c})")
    print(f"Binary region: rows {sample_r-1}..{sample_r+4}, cols {sample_c-1}..{sample_c+4}")
    print(f"Dilated region: rows {sample_r-1}..{sample_r+4}, cols {sample_c-1}..{sample_c+4}")

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('images/Shapes1.bmp')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold - foreground=1, background=0
_, binary = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY)

# Kernel for a filled 4x4 square:
# Inner 4x4 = foreground (1), outer frame = background (0) → total 6x6
hmt_kernel = np.array([
    [ 0,  0,  0,  0,  0,  0],
    [ 0,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  0],
    [ 0,  0,  0,  0,  0,  0]
], dtype=np.int8)

# NOTE 1: In HMT, background should be specified with -1
hmt_kernel_correct = np.array([
    [-1, -1, -1, -1, -1, -1],
    [-1,  1,  1,  1,  1, -1],
    [-1,  1,  1,  1,  1, -1],
    [-1,  1,  1,  1,  1, -1],
    [-1,  1,  1,  1,  1, -1],
    [-1, -1, -1, -1, -1, -1]
], dtype=np.int8)

# NOTE 2: anchor=(-1,-1) is center (default), no foreground inversion
hit_miss = cv2.morphologyEx(
    binary.astype(np.uint8),
    cv2.MORPH_HITMISS,
    hmt_kernel_correct,
    anchor=(-1, -1)   # <-- düzeltildi
)

# Print detected square locations
locs = np.argwhere(hit_miss == 1)
print(f"Detected squares: {len(locs)}")
for loc in locs:
    print(f"  Square location: row={loc[0]}, col={loc[1]}")

# Visualize
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Hit-or-Miss (only squares)')
plt.imshow(hit_miss, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
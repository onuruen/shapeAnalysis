import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('images/Shapes1.bmp')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
binary_01 = (binary // 255).astype(np.uint8)

hmt_kernel = np.array([
    [-1, -1, -1, -1, -1, -1],
    [-1,  1,  1,  1,  1, -1],
    [-1,  1,  1,  1,  1, -1],
    [-1,  1,  1,  1,  1, -1],
    [-1,  1,  1,  1,  1, -1],
    [-1, -1, -1, -1, -1, -1]
], dtype=np.int8)

# anchor=(1,1) → hit noktası karenin sol üst köşesine düşer
hit_miss = cv2.morphologyEx(binary_01, cv2.MORPH_HITMISS, hmt_kernel, anchor=(1, 1))
locs = np.argwhere(hit_miss == 1)
print(f"Detected squares: {len(locs)}")

# cv2.dilate anchor offset sorunu var → manuel dilation (garantili)
dilated = np.zeros_like(binary_01)
for loc in locs:
    r, c = loc  # sol üst köşe
    dilated[r:r+4, c:c+4] = 1

dilated_255 = (dilated * 255).astype(np.uint8)

# Image Difference
difference = cv2.subtract(binary, dilated_255)

plt.figure(figsize=(20, 5))
plt.subplot(1, 4, 1)
plt.title('Original (Binary)')
plt.imshow(binary, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title('Hit-or-Miss (kareler)')
plt.imshow(hit_miss * 255, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title('Dilated (geri kazanılan kareler)')
plt.imshow(dilated_255, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title('Difference (L + hollow)')
plt.imshow(difference, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
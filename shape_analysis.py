import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('images/Shapes1.bmp')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
binary_01 = (binary // 255).astype(np.uint8)

# ── SECTION 1: SQUARE DETECTION AND ELIMINATION ─────────────────────────────
hmt_square_kernel = np.array([
    [-1, -1, -1, -1, -1, -1],
    [-1,  1,  1,  1,  1, -1],
    [-1,  1,  1,  1,  1, -1],
    [-1,  1,  1,  1,  1, -1],
    [-1,  1,  1,  1,  1, -1],
    [-1, -1, -1, -1, -1, -1]
], dtype=np.int8)

hit_squares = cv2.morphologyEx(binary_01, cv2.MORPH_HITMISS, hmt_square_kernel, anchor=(1, 1))
square_locations = np.argwhere(hit_squares == 1)
print(f"Detected squares: {len(square_locations)}")

dilated_squares = np.zeros_like(binary_01)
for loc in square_locations:
    r, c = loc
    dilated_squares[r:r+4, c:c+4] = 1
dilated_squares_255 = (dilated_squares * 255).astype(np.uint8)
diff_no_squares = cv2.subtract(binary, dilated_squares_255)
diff_no_squares_01 = (diff_no_squares // 255).astype(np.uint8)

# ── SECTION 2: L-SHAPE DETECTION AND ELIMINATION ──────────────────────────────
hmt_l_kernel = np.array([
    [ 1,  1,  1,  1,  1,  1,  1],
    [ 1,  1,  1,  1,  1,  1,  1],
    [ 1,  1,  1,  1,  1,  1,  1],
    [ 1,  1,  1,  1,  1,  1,  1],
    [ 1,  1,  1,  1, -1, -1, -1],
    [ 1,  1,  1,  1, -1, -1, -1],
], dtype=np.int8)

hit_l_shapes = cv2.morphologyEx(diff_no_squares_01, cv2.MORPH_HITMISS, hmt_l_kernel, anchor=(0, 0))
l_shape_locations = np.argwhere(hit_l_shapes == 1)
print(f"Detected L-shapes: {len(l_shape_locations)}")

l_shape_mask = np.ones((6, 7), dtype=np.uint8)
l_shape_mask[4:6, 4:7] = 0

recovered_l_shapes = np.zeros_like(binary_01)
for loc in l_shape_locations:
    r, c = loc
    recovered_l_shapes[r:r+6, c:c+7] = np.maximum(recovered_l_shapes[r:r+6, c:c+7], l_shape_mask)
recovered_l_shapes_255 = (recovered_l_shapes * 255).astype(np.uint8)
diff_no_l_shapes = cv2.subtract(binary, recovered_l_shapes_255)

# ── FIGURE 1: SQUARE ANALYSIS ─────────────────────────────────────────────────
fig1, axes1 = plt.subplots(1, 4, figsize=(20, 5))
fig1.suptitle('Square Analysis', fontsize=14, fontweight='bold')

axes1[0].imshow(binary, cmap='gray')
axes1[0].set_title('Original (Binary)')
axes1[0].axis('off')

axes1[1].imshow(hit_squares * 255, cmap='gray')
axes1[1].set_title('HMT Square Detection')
axes1[1].axis('off')

axes1[2].imshow(dilated_squares_255, cmap='gray')
axes1[2].set_title('Dilated Squares')
axes1[2].axis('off')

axes1[3].imshow(diff_no_squares, cmap='gray')
axes1[3].set_title('Difference: L-shapes + Hollow Circles')
axes1[3].axis('off')

plt.tight_layout()

# ── FIGURE 2: L-SHAPE ANALYSIS ────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 4, figsize=(20, 5))
fig2.suptitle('L-shape Analysis', fontsize=14, fontweight='bold')

axes2[0].imshow(binary, cmap='gray')
axes2[0].set_title('Original (Binary)')
axes2[0].axis('off')

axes2[1].imshow(hit_l_shapes * 255, cmap='gray')
axes2[1].set_title('HMT L-shape Detection')
axes2[1].axis('off')

axes2[2].imshow(recovered_l_shapes_255, cmap='gray')
axes2[2].set_title('Dilated L-shapes')
axes2[2].axis('off')

axes2[3].imshow(diff_no_l_shapes, cmap='gray')
axes2[3].set_title('Difference: Squares and Hollow Circles')
axes2[3].axis('off')


# ── SECTION 3: HOLLOW CIRCLES ONLY ───────────────────────────────────────────
hollow_only = cv2.subtract(diff_no_squares, recovered_l_shapes_255)

# ── FIGURE 3: HOLLOW CIRCLE ANALYSIS ─────────────────────────────────────────
fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
fig3.suptitle('Hollow Circle Extraction', fontsize=14, fontweight='bold')

axes3[0].imshow(diff_no_squares, cmap='gray')
axes3[0].set_title('Hollow + L-shapes')
axes3[0].axis('off')

axes3[1].imshow(recovered_l_shapes_255, cmap='gray')
axes3[1].set_title('L-shapes Mask (to subtract)')
axes3[1].axis('off')

axes3[2].imshow(hollow_only, cmap='gray')
axes3[2].set_title('Hollow Circles Only')
axes3[2].axis('off')

plt.tight_layout()
plt.show()
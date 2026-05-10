import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('images/Shapes1.bmp')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
binary_01 = (binary // 255).astype(np.uint8)

# ── BÖLÜM 1: KARE TESPİTİ VE ELEMESİ ──────────────────────────────────────
hmt_square = np.array([
    [-1, -1, -1, -1, -1, -1],
    [-1,  1,  1,  1,  1, -1],
    [-1,  1,  1,  1,  1, -1],
    [-1,  1,  1,  1,  1, -1],
    [-1,  1,  1,  1,  1, -1],
    [-1, -1, -1, -1, -1, -1]
], dtype=np.int8)

hit_squares = cv2.morphologyEx(binary_01, cv2.MORPH_HITMISS, hmt_square, anchor=(1, 1))
square_locs = np.argwhere(hit_squares == 1)
print(f"Detected squares: {len(square_locs)}")

dilated_squares = np.zeros_like(binary_01)
for loc in square_locs:
    r, c = loc
    dilated_squares[r:r+4, c:c+4] = 1
dilated_squares_255 = (dilated_squares * 255).astype(np.uint8)
diff_no_squares = cv2.subtract(binary, dilated_squares_255)
diff_no_squares_01 = (diff_no_squares // 255).astype(np.uint8)

# ── BÖLÜM 2: L-ŞEKİL TESPİTİ VE ELEMESİ ───────────────────────────────────
hmt_L = np.array([
    [ 1,  1,  1,  1,  1,  1,  1],
    [ 1,  1,  1,  1,  1,  1,  1],
    [ 1,  1,  1,  1,  1,  1,  1],
    [ 1,  1,  1,  1,  1,  1,  1],
    [ 1,  1,  1,  1, -1, -1, -1],
    [ 1,  1,  1,  1, -1, -1, -1],
], dtype=np.int8)

hit_L = cv2.morphologyEx(diff_no_squares_01, cv2.MORPH_HITMISS, hmt_L, anchor=(0, 0))
L_locs = np.argwhere(hit_L == 1)
print(f"Detected L-shapes: {len(L_locs)}")

L_mask = np.ones((6, 7), dtype=np.uint8)
L_mask[4:6, 4:7] = 0

recovered_L = np.zeros_like(binary_01)
for loc in L_locs:
    r, c = loc
    recovered_L[r:r+6, c:c+7] = np.maximum(recovered_L[r:r+6, c:c+7], L_mask)
recovered_L_255 = (recovered_L * 255).astype(np.uint8)
diff_no_L = cv2.subtract(binary, recovered_L_255)

# ── FİGÜR 1: KARE ANALİZİ ───────────────────────────────────────────────────
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
axes1[3].set_title('Diff: Only L + Hollow')
axes1[3].axis('off')

plt.tight_layout()

# ── FİGÜR 2: L-ŞEKİL ANALİZİ ───────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 4, figsize=(20, 5))
fig2.suptitle('L-shape Analysis', fontsize=14, fontweight='bold')

axes2[0].imshow(binary, cmap='gray')
axes2[0].set_title('Original (Binary)')
axes2[0].axis('off')

axes2[1].imshow(hit_L * 255, cmap='gray')
axes2[1].set_title('HMT L-shape Detection')
axes2[1].axis('off')

axes2[2].imshow(recovered_L_255, cmap='gray')
axes2[2].set_title('Dilated L-shapes')
axes2[2].axis('off')

axes2[3].imshow(diff_no_L, cmap='gray')
axes2[3].set_title('Diff: Only Square + Hollow')
axes2[3].axis('off')

plt.tight_layout()
plt.show()
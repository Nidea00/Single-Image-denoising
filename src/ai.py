import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.restoration import estimate_sigma
from scipy.ndimage import sobel

def gradient(u):
    ux = np.roll(u, -1, axis=1) - u
    uy = np.roll(u, -1, axis=0) - u
    return ux, uy

def divergence(px, py):
    fx = px - np.roll(px, 1, axis=1)
    fy = py - np.roll(py, 1, axis=0)
    return fx + fy

def shrink(x, thresh):
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)

def split_bregman_tv_denoise(f, lam=2.0, mu=5.0, max_iter=100):
    u = f.copy()
    dx = np.zeros_like(f)
    dy = np.zeros_like(f)
    bx = np.zeros_like(f)
    by = np.zeros_like(f)

    for i in range(max_iter):
        # Step 1: Solve for u (Poisson-like equation)
        ux, uy = gradient(u)
        dx_bar = dx - bx
        dy_bar = dy - by
        div_d_b = divergence(dx_bar, dy_bar)

        u = (lam * f + mu * div_d_b) / (lam + 4 * mu)

        # Step 2: Update d (shrinkage)
        ux, uy = gradient(u)
        dx = shrink(ux + bx, 1.0 / mu)
        dy = shrink(uy + by, 1.0 / mu)

        # Step 3: Update b
        bx = bx + ux - dx
        by = by + uy - dy

    return u

# 測試用圖片
image = img_as_float(data.camera())

noisy = image + 0.1 * np.random.randn(*image.shape)
noisy = np.clip(noisy, 0, 1)

# 執行 Split Bregman 去噪
denoised = split_bregman_tv_denoise(noisy, lam=2.0, mu=5.0, max_iter=50)

# 顯示結果
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Noisy")
plt.imshow(noisy, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Denoised (Split Bregman)")
plt.imshow(denoised, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()

# It uses the split Bregman method to minimize the Rudin-Osher-Fatemi model
# Find u, d to minimize |d| + lambda * |u-f|^2 /2 s.t. d = gradi(u).

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float

def psnr(img1, img2):
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0/mse)

class rof:
    def __init__(self, img):
        self.image = img
        
    def gradient(self, u):
        ux = np.roll(u, -1, axis=1) - u
        uy = np.roll(u, -1, axis=0) - u
        return ux, uy
    
    def divergence(selft, px, py):
        fx = px - np.roll(px, 1, axis=1)
        fy = py - np.roll(py, 1, axis=0)
        return fx + fy
    
    def shrink(self, x, thresh):
        return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)
    
    def set(self, l = 2.0, mu = 5.0, TOL = 1e-4, max_iter = 100):
        info = {"lambda": l, "mu": mu, "TOL": TOL, "max_iter":max_iter}
        
        self.iterative_info = info
    
    def split_Bregman(self, lam = 2.0, mu=5.0, TOL = 1e-4, max_iter=100):
        u = self.image.copy()
        dx = np.zeros_like(self.image)
        dy = np.zeros_like(self.image)
        bx = np.zeros_like(self.image)
        by = np.zeros_like(self.image)
        
        for _ in range(max_iter):
            u0 = u.copy()
            
            # u-subproblem
            ux, uy = self.gradient(u)
            div_d_b = self.divergence(dx - bx, dy-by)
            
            u = (lam * self.image + mu * div_d_b) / (lam + 4 * mu)
            
            # d-subproblem
            ux, uy = self.gradient(u)
            dx = self.shrink(ux + bx, 1.0/mu)
            dy = self.shrink(uy + by, 1.0/mu)
            
            # Updata b
            bx += ux - dx
            by += uy - dy
            
            diff = np.linalg.norm(u - u0) / np.linalg.norm(u0)
            if diff < TOL: break 
        
        return u

if __name__ == "__main__":
    
    images_folder_path = "images/"
    original_image_name = "lena.png"
    noisy_image_name = "noisy_lena.png"
    
    l = 10.0
    
    noisy_image = img_as_float(io.imread(images_folder_path + noisy_image_name))
    origin_image = img_as_float(io.imread(images_folder_path + original_image_name))
    
    ''' image data.camera()
    from skimage import data
    origin_image = img_as_float(data.camera())
    noisy_image = origin_image + 0.1 * np.random.randn(*origin_image.shape)
    noisy_image = np.clip(noisy_image, 0, 1)
    '''
    
    model = rof(noisy_image)
    denoised_image = model.split_Bregman(lam=l, mu=2.5, TOL=1e-4, max_iter=100)
    
    # print the images
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(origin_image, cmap="gray")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Noisy image")
    plt.imshow(noisy_image, cmap="gray")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Denoised image with lambda = " + str(l))
    plt.imshow(denoised_image, cmap="gray")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
# It uses the split Bregman method to minimize the Rudin-Osher-Fatemi model
# Find u, d to minimize |d| + lambda * |u-f|^2 /2 s.t. d = gradi(u).

import numpy as np

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
    
    def project_onto_unit_ball(self, px, py):
        norm = np.maximum(1.0, np.sqrt(px**2 + py**2))
        return px / norm, py / norm
    
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
        
    def chambolle_pock(self, lam = 2.0, tau = 0.25, sigma= 0.25, theta = 1.0, TOL = 1e-4, max_iter = 100):
        u = self.image.copy()
        u_bar = u.copy()
        
        px = np.zeros_like(self.image)
        py = np.zeros_like(self.image)
        
        for _ in range(max_iter):
            ux_bar, uy_bar = self.gradient(u_bar)
            px += sigma * ux_bar
            py += sigma * uy_bar
            px, py = self.project_onto_unit_ball(px, py)
            
            div_p = self.divergence(px, py)
            
            u0 = u.copy()
            u = (u + tau * lam * (self.image + div_p)) / (1 + tau * lam)
            
            u_bar = u + theta * (u - u0)
            
            diff = np.linalg.norm(u - u0) / np.linalg.norm(u0)
            if diff < TOL: break
        
        return u
        
    def low_pass_filter(self, shape, radius):
        h, w = shape
        y, x = np.ogrid[:h, :w]
        cy, cx = h // 2, w // 2
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        return (dist <= radius).astype(float)
        
    
    def fft(self):
        self.img_fft = np.fft.fft2(self.image)
        self.img_fftshift = np.fft.fftshift(self.img_fft)
        
    
    def fft_denoise(self, radius = 30):
        self.fft()
        
        mask = self.low_pass_filter(self.image.shape, radius)
        fshift_filtered = self.img_fftshift * mask
        f_ishift = np.fft.ifftshift(fshift_filtered)
        
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        return img_back
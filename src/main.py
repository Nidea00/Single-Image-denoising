import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float

from rof import rof

def save_as_png(gray, path):
    
    min_val = np.min(gray)
    max_val = np.max(gray)
    
    if min_val == max_val:
        gray_uint8 = np.zeros_like(gray, dtype=np.uint8)
    else:
        gray_uint8 = ((gray - min_val)/(max_val - min_val) * 255).astype(np.uint8)
    
    io.imsave(path, gray_uint8)

if __name__ == "__main__" :
    # The path of images
    origin_image_path = "images/lena.png"
    noisy_image_path = "images/noisy_lena.png"
    
    output_folder_path = "output/"
    
    #
    
    # Read the original and the noisy image
    origin_img = io.imread(origin_image_path)
    origin_img = img_as_float(origin_img)
    noisy_img = io.imread(noisy_image_path)
    noisy_img = img_as_float(noisy_img)
    
    # Call function
    model = rof(noisy_img)
    denoised_img_sb = model.split_Bregman(lam = 5.0)
    denoised_img_cp = model.chambolle_pock(lam = 2.5)
    denoised_img_fft = model.fft_denoise(radius = 50)
    
    
    # Transform to uint8 and save
    save_as_png(denoised_img_sb, output_folder_path + "denoised_sb.png")
    save_as_png(denoised_img_cp, output_folder_path + "denoised_cp.png")
    save_as_png(denoised_img_fft, output_folder_path + "denoised_fft.png")
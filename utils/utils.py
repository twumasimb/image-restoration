
import os
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
import glob
import shutil
import os
import matplotlib.pyplot as plt


#Function to evaluate model
def model_eval(raw_dir, gen_dir):
    # Image paths
    total_psnr = []
    total_ssim = []
    for raw in os.listdir(raw_dir):
        for gen in os.listdir(gen_dir):
            if(raw == gen):

                # Load the two images
                raw_img = cv2.imread(os.path.join(raw_dir, raw))
                gen_img = cv2.imread(os.path.join(gen_dir, gen))

                # Get the dimensions of the images
                height1, width1, channels1 = raw_img.shape
                height2, width2, channels2 = gen_img.shape

                #Printing the shapes
                # print(f"restored Image: {gen_img.shape}")
                # print(f"Input Image: {raw_img.shape}")

                # Determine the maximum dimensions
                max_width = max(width1, width2)
                max_height = max(height1, height2)

                # Resize the images to have the same dimensions
                raw_img_resized = cv2.resize(raw_img, (max_width, max_height))
                gen_img_resized = cv2.resize(gen_img, (max_width, max_height))

                # Compute the PSNR between the two images
                psnr = peak_signal_noise_ratio(raw_img_resized, gen_img_resized)

                # Compute the SSIM between the two images
                ssim = structural_similarity(raw_img_resized, gen_img_resized, multichannel=True, win_size=3, channel_axis=2)

                total_psnr.append(psnr)
                total_ssim.append(ssim)

                # print("PSNR:", psnr)
                # print("SSIM:", ssim)
    # Compute the average psnr and average ssim
    avg_psnr = sum(total_psnr)/len(total_psnr)
    avg_ssim = sum(total_ssim)/len(total_ssim)

    return avg_psnr, avg_ssim

#Function to copy 
def copyImages(src, dst, size):
    k = 1
    for pngfile in glob.iglob(os.path.join(src, "*.jpg")):
        if(k <= size):
            shutil.copy(pngfile, dst)
            k+=1
        else:
            continue

def load_img(filepath):
    img = plt.imread(filepath)
    # img = img.astype(np.float32)
    # img = img/255.
    return img

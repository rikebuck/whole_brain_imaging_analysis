bin_imgs_folder = "/Volumes/Extreme SSD/saved_imgs/2022-06-09-01/bin_imgs/"
xs_csv = "/Volumes/Extreme SSD/saved_imgs/2022-06-09-01/wormtracer_results/2024-08-22 09_36_10.497589_x.csv"
ys_csv = "/Volumes/Extreme SSD/saved_imgs/2022-06-09-01/wormtracer_results/2024-08-22 09_36_10.497589_y.csv"
save_dir = "/Volumes/Extreme SSD/saved_imgs/2022-06-09-01/wormtracer_results/"
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import glob
import os
import cv2
import sys
sys.path.append('/Users/friederikebuck/Desktop/WormTracking/LargePlateWormTracker/')
from FileHandlers.VideoHandlers.ImgVideoHandler import FrameVideoHandler
pnghandler = FrameVideoHandler(bin_imgs_folder, "*.png", is_compressed = False)
sorted_img_file_names = pnghandler.get_sorted_img_file_names()
midline_xs = pd.read_csv(xs_csv).to_numpy()
midline_ys = pd.read_csv(ys_csv).to_numpy()
for i, img_file in enumerate(sorted_img_file_names):
    img = cv2.imread(img_file)

    # Convert the image to RGB (since OpenCV loads images in BGR by default)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Assuming midline_xs and midline_ys are numpy arrays with coordinates
    for x, y in zip(midline_xs[i+1, :], midline_ys[i+1, :]):
        # Draw a small circle at each (x, y) coordinate
        cv2.circle(img_rgb, (int(x), int(y)), radius=3, color=(255, 0, 0), thickness=-1)

    # Save the result
    cv2.imwrite(os.path.join(save_dir, str(i)+'_midline.png'), img_rgb)
    if i == len(sorted_img_file_names)-2:
        break

    
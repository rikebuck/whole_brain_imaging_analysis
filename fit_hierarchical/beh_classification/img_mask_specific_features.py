import os
import cv2
import numpy as np
from get_data.flavell_raw_nir_extraction.load_raw_frames import get_is_looping_from_mask
from get_data.flavell_raw_nir_extraction.convert_nir_to_confocal import convert_nir_to_confocal
def get_is_looping_from_mask_dir_nir(n_frames, bin_img_dir, area_to_perimeter_ratio_thres = 1, area_to_perimeter = 6):
    is_looping = []
    a_to_ps = []
    areas = []
    for frame in range(n_frames):
        img = cv2.imread(os.path.join(bin_img_dir, str(frame+1).zfill(5)+".png"))[:, :, 0]
        looping, a_to_p, area =  get_is_looping_from_mask(img,area_to_perimeter_ratio_thres= area_to_perimeter_ratio_thres,  area_to_perimeter = area_to_perimeter)
        is_looping.append(looping)
        a_to_ps.append(a_to_p)
        areas.append(area)
    # print(a_to_ps)
    #plt.hist(a_to_ps, bins = 100)
    a_to_ps = np.array(a_to_ps)
    np.argwhere(np.abs(a_to_ps-6.3)<0.5).flatten()
    return np.array(is_looping), np.array(a_to_ps), np.array(areas)

def get_is_looping_from_mask_dir_confocal(n_frames_nir, 
                                          n_frames_confocal
                                          ,nir_to_confocal, bin_img_dir, area_to_perimeter_ratio_thres = 1, area_to_perimeter = 6):
    is_looping_bin_img, a_to_ps, areas = get_is_looping_from_mask_dir_nir(n_frames_nir, bin_img_dir, area_to_perimeter_ratio_thres= area_to_perimeter_ratio_thres,  area_to_perimeter = area_to_perimeter)
    is_looping_bin_img_confocal = convert_nir_to_confocal(nir_to_confocal,#[1:],
                                                          is_looping_bin_img)[1:]
    is_looping_bin_img_confocal = np.isin(np.arange(0,n_frames_confocal),np.argwhere(is_looping_bin_img_confocal).flatten())
    return is_looping_bin_img_confocal, a_to_ps,areas
 
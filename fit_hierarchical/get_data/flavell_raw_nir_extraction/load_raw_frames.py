import h5py
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt

import sys 
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('/Users/friederikebuck/Desktop/WormTracking/LargePlateWormTracker/')
from FileHandlers.OSHandler import OSHandler
from PIL import Image

from skimage import filters, measure
import cv2 as cv2
import copy 
def get_gaussian_kernel(k): #https://stackoverflow.com/questions/61394826/how-do-i-get-to-show-gaussian-kernel-for-2d-opencv
    img=np.zeros((k+2,k+2))
    center = np.floor([(k+2)/2,(k+2)/2]).astype('int')
    img[center[0], center[1]]=1
    kern = cv2.GaussianBlur(img, (k, k), 0)
    return kern


# edges_sobel = filters.sobel(image)
# threshold_value = 0.1  # You can adjust this value
# binary_sobel = edges_sobel > threshold_value
# plt.imshow(binary_sobel, cmap='gray')

# # Find contours in the binary image
# contours = measure.find_contours(binary_sobel, level=0.8)

# # Identify the largest contour by area
# largest_contour = max(contours, key=lambda x: cv2.contourArea(np.array(x, dtype=np.int32)))


# # Create an empty mask
# mask = np.zeros_like(binary_sobel)

# # Draw the largest contour on the mask
# rr, cc = np.array(largest_contour, dtype=np.int32).T
# mask[rr, cc] = 1

# plt.imshow(mask*255)


# ###okay###
# blur_image1= cv2.medianBlur(img,15)
# # blurred_image = cv2.GaussianBlur(blur_image1, (11, 11), 0)
# edges_sobel = filters.sobel(blur_image1)
# plt.imshow(blur_image1)
# fig, ax = plt.subplots()
# plt.imshow(edges_sobel)
# threshold_value = 0.01  # You can adjust this value
# binary_sobel = edges_sobel > threshold_value
# binary_sobel = binary_sobel.astype('int8')*255
# fig,ax = plt.subplots()
# plt.imshow(binary_sobel)
# # Create an empty mask
# mask = np.zeros_like(binary_sobel)
# # Find contours in the binary image


# contours = measure.find_contours(binary_sobel, level=0.8)
# # Identify the largest contour by area
# largest_contour = max(contours, key=lambda x: cv2.contourArea(np.array(x, dtype=np.int32)))
# # Draw the largest contour on the mask
# rr, cc = np.array(largest_contour, dtype=np.int32).T
# mask[rr, cc] = 1

# fig,ax = plt.subplots()
# ax.imshow(mask)





# fig,ax = plt.subplots()
# kernel = np.ones((5,5))


# # fig,ax = plt.subplots()
# # plt.imshow(cv2.dilate(binary_sobel,kernel,iterations = 1))


# # fig,ax = plt.subplots()
# # plt.imshow(cv2.dilate(binary_sobel,kernel,iterations = 1))


# fig,ax = plt.subplots()
# s2 = cv2.GaussianBlur(binary_sobel, (11, 11), 0)
# plt.imshow(s2)
# # kernel = np.ones((5,5))
# # closing = cv2.morphologyEx(s2, cv2.MORPH_CLOSE, kernel)
# # plt.imshow(closing)

# # Find contours in the binary image

# contours = measure.find_contours(s2, level=0.8)
# # Identify the largest contour by area
# largest_contour = max(contours, key=lambda x: cv2.contourArea(np.array(x, dtype=np.int32)))


# # Create an empty mask
# mask = np.zeros_like(binary_sobel)

# # Draw the largest contour on the mask
# rr, cc = np.array(largest_contour, dtype=np.int32).T
# mask[rr, cc] = 1
# mask = cv2.fillPoly(mask, [np.array(largest_contour, dtype=np.int32)], 1)
# fig,ax = plt.subplots()
# ax.imshow(mask)

# # kernel = get_gaussian_kernel(29)#np.ones((30, 30))
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
# opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
# fig,ax = plt.subplots()
# ax.imshow(opening)




# # # Find contours in the binary image
# # contours = measure.find_contours(binary_sobel, level=0.8)

# # # Identify the largest contour by area
# # largest_contour = max(contours, key=lambda x: cv2.contourArea(np.array(x, dtype=np.int32)))
# # mask = cv2.fillPoly(mask, [np.array(largest_contour, dtype=np.int32)], 1)

# # fig,ax = plt.subplots()
# # ax.imshow(mask)


def resize_img(image, rescale_factor = 0.2):
    height, width = image.shape[:2]
    new_width = int(width * rescale_factor)
    new_height = int(height * rescale_factor)
    new_dimensions = (new_width, new_height)

    # Resize the image
    resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)
    return resized_image

def get_img_mask(image, sobel_thres = 0.01):
    #blur image to remove small edges 
    blur_image1= cv2.medianBlur(image,15)
    edges_sobel = filters.sobel(blur_image1)
    binary_sobel = edges_sobel > sobel_thres
    binary_sobel = binary_sobel.astype('int8')*255
    
    
    s2 = cv2.GaussianBlur(binary_sobel, (11, 11), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    #a = cv2.dilate(s2,kernel,iterations = 1)
    
    a =  cv2.morphologyEx(s2, cv2.MORPH_CLOSE, kernel)
    plt.imshow(a)
    # fig,ax = plt.subplots()
    # plt.imshow(s2)
    
    contours = measure.find_contours(a, level=0.8)
    largest_contour = max(contours, key=lambda x: cv2.contourArea(np.array(x, dtype=np.int32)))
    largest_contour_flipped = np.array(largest_contour, dtype=np.int32)[:, ::-1]
    rr, cc = np.array(largest_contour, dtype=np.int32).T
    mask = np.zeros_like(binary_sobel)
    mask[rr, cc] = 1
    mask = cv2.fillPoly(mask, [np.array(largest_contour_flipped, dtype=np.int32)], 1)
    
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
    fig,ax = plt.subplots()
    ax.imshow(opening)
    
    masked_image = opening * image
    plt.imshow(masked_image)
    
    
    return opening, masked_image, 

def convert_contour(contour):
    return np.array(contour, dtype=np.int32).reshape(-1, 1, 2)

def get_is_looping(outer_worm_contour,  area_to_perimeter_ratio_thres=1.15,  area_to_perimeter = 22.5):
   
    if cv2.contourArea(outer_worm_contour)/cv2.arcLength(outer_worm_contour,True)>area_to_perimeter*area_to_perimeter_ratio_thres:

        return True, cv2.contourArea(outer_worm_contour)/cv2.arcLength(outer_worm_contour,True), cv2.contourArea(outer_worm_contour)
    else:
        # print("not looping")
        # print("ratio", cv2.contourArea(self.outer_worm_contour)/cv2.arcLength(self.outer_worm_contour,True))
        return False , cv2.contourArea(outer_worm_contour)/cv2.arcLength(outer_worm_contour,True), cv2.contourArea(outer_worm_contour)

def get_is_looping_from_mask(mask,area_to_perimeter_ratio_thres = 1.15, area_to_perimeter = 22.5):

    contours = measure.find_contours(mask, level=0.8)

    # Convert contours to numpy arrays
    contours = [np.array(contour, dtype=np.int32) for contour in contours]

    # Sort contours by area
    contours.sort(key=lambda x: cv2.contourArea(x), reverse=True)

    # The largest contour is assumed to be the outer contour
    largest_contour = convert_contour(copy.deepcopy(contours[0]))
    largest_contour_flipped = contours[0][:, ::-1]

    # Determine if there are inner contours
    ##check if is looping and only do this if is looping: 
    is_looping, a_to_p, area = get_is_looping(largest_contour, area_to_perimeter_ratio_thres= area_to_perimeter_ratio_thres,  area_to_perimeter =  area_to_perimeter )
    return is_looping, a_to_p, area

if __name__ == "__main__":
    import glob
    from pathlib import Path

    date_strs= [
    # "2022-07-19", 
    # "2022-07-20", 
    # "2022-07-26",
    # "2022-08-02", ##todo!!
    # "2022-12-21", 
    # "2023-01-06", 
    # "2023-01-09", 
    # "2023-01-10", 
    # "2023-01-13", 

    "2023-01-19",
    # "2023-01-23", 
    # "2023-01-24", 

    # "2022-07-29", #####didnt finish!!!
    "2023-01-17",  #####didnt finish!!!


    ]

    # # for date_str in date_strs: 
    #     file_paths = glob.glob(os.path.join("/Volumes/Extreme SSD/data/raw_nir_frames/",date_str, date_str+"*.h5"))
    #     print("file_paths", file_paths)
    file_paths = ["/Volumes/Extreme SSD/data/raw_nir_frames/2023-01-17/2023-01-17-01.h5"]
    for file_path in file_paths:
        basename = Path(file_path).stem
        print(basename)
        print(file_path)

        saved_img_folder = "/Volumes/Extreme SSD/saved_imgs/"+basename+"/"
        masked_img_folder = OSHandler().make_new_folder(saved_img_folder, "masked_imgs")
        bin_img_folder = OSHandler().make_new_folder(saved_img_folder, "bin_imgs")
        print("saved_img_folder", saved_img_folder)

        with h5py.File(file_path, 'r') as file:
            # List contents of the file
            subtypes = list(file.keys())
            print("subtypes", subtypes)
            imgs = file["img_nir"][()]

        for i in range(imgs.shape[0]): 
            if i%500 == 0: 
                print("i", i)
            img = imgs[i, :, :]
            ###
            ##
            image = img
            sobel_thres = 0.01
            min_pt_threshold = 90 
                #blur image to remove small edges 
            blur_image1 = cv2.medianBlur(image,15)
            # blur_image1 = image
            edges_sobel = filters.sobel(blur_image1)
            binary_sobel = edges_sobel > sobel_thres
            binary_sobel = binary_sobel.astype('int8')*255
            
            
            s2 = cv2.GaussianBlur(binary_sobel, (11, 11), 0)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
            #
            
            a =  cv2.morphologyEx(s2, cv2.MORPH_CLOSE, kernel)
            # plt.imshow(a)

            contours = measure.find_contours(a, level=0.8)

            # Convert contours to numpy arrays
            contours = [np.array(contour, dtype=np.int32) for contour in contours]

            # Sort contours by area
            contours.sort(key=lambda x: cv2.contourArea(x), reverse=True)

            # The largest contour is assumed to be the outer contour
            largest_contour = convert_contour(copy.deepcopy(contours[0]))
            largest_contour_flipped = contours[0][:, ::-1]
            mask = np.zeros_like(a, dtype=np.uint8)
            mask = cv2.fillPoly(mask, [largest_contour_flipped], 1)

            # Determine if there are inner contours
            ##check if is looping and only do this if is looping: 
            is_looping, a_to_p, a = get_is_looping(largest_contour)
            if is_looping: 
                print("is looping frame", i , a_to_p)
                
                inner_contours = [contour for contour in contours[1:] if contour.shape[0] > min_pt_threshold]

                inner_contour_shapes = np.array([cv2.contourArea(cnt) for cnt in contours[1:] ])
                if  len(inner_contours)>0: 
                    inner_contour_i = np.argmax(inner_contour_shapes)
                    inner_contour = inner_contours[inner_contour_i]
                    # For each inner contour, subtract from the mask to get the area between contours
                    
                    inner_contour_flipped = inner_contour[:, ::-1]
                    mask_inner = np.zeros_like(a, dtype=np.uint8)
                    mask_inner = cv2.fillPoly(mask_inner, [inner_contour_flipped], 1)
                    mask = mask - mask_inner
                

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            
            # fig,ax = plt.subplots()
            # plt.imshow(mask)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
            opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            # fig,ax = plt.subplots()
            # ax.imshow(opening)
            contours = measure.find_contours(opening, level=0.8)
            largest_contours = [cnt for cnt in contours if cnt.shape[0]>100]

            # if len(contours)>1 and not is_looping:
            #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
            #     closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
            #     s = closing
            #     opening = s
            # else: 
            #     s = opening
            s = opening
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
            opening = cv2.morphologyEx(s, cv2.MORPH_OPEN, kernel)
            
            
            ###remove any excess contours 
            contours = measure.find_contours(opening, level=0.8)
            bin_img = np.zeros_like(opening, dtype=np.uint8)
            contours = [convert_contour(cnt[:, ::-1]) for cnt in contours]
            contours.sort(key=lambda x: cv2.contourArea(x), reverse=True)
            largest_contour = convert_contour(copy.deepcopy(contours[0]))
            largest_contour_flipped = contours[0]
            bin_img = cv2.fillPoly(bin_img, [largest_contour_flipped], 1)
            
            if is_looping: 
                inner_contours = [contour for contour in contours[1:] if contour.shape[0] > min_pt_threshold]

                inner_contour_shapes = np.array([cv2.contourArea(cnt) for cnt in contours[1:] ])
                if inner_contour_shapes.shape[0]>0 and len(inner_contours)>0: 
                    inner_contour_i = np.argmax(inner_contour_shapes)
                    inner_contour = inner_contours[inner_contour_i]
                    # For each inner contour, subtract from the mask to get the area between contours
                    
                    inner_contour_flipped = inner_contour[:, ::-1]
                    mask_inner = np.zeros_like(a, dtype=np.uint8)
                    mask_inner = cv2.fillPoly(mask_inner, [inner_contour_flipped], 1)
                    bin_img = bin_img - mask_inner
            
            
            
            
            masked_image = bin_img * image
            bin_img = bin_img*255
            
            # plt.imshow(masked_image)
            ####
            ####
            # bin_img, masked_img = get_img_mask(img, sobel_thres = 0.01)
            
            masked_image = resize_img(masked_image, rescale_factor = 0.2)
            masked_image_pil = Image.fromarray(masked_image)
            masked_image_pil.save(os.path.join(masked_img_folder, str(i).zfill(5)+".png"))
            
            bin_img = resize_img(bin_img, rescale_factor = 0.2)
            bin_img_pil = Image.fromarray(bin_img)
            bin_img_pil.save(os.path.join(bin_img_folder, str(i).zfill(5)+".png"))
            
        

    print()
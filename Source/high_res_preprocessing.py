import os 
from skimage import io
from image_display import display_image
import file
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import closing
from skimage.measure import label, regionprops
import numpy as np
import scipy.ndimage
from scipy.ndimage import rotate

c_w = 250 # set cropped image width
num_cells = 0

def get_high_res_image(name):
# =============================================================================
#     Gets a high res (60X) image from the "High_Res_Input_Images" folder
#     and returns it as a matrix
# =============================================================================
    
    parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    image_path = os.path.join(parent, "High_Res_Input_Images", name)
    image = io.imread(image_path)        
    return image

def rotateImage(img, angle, pivot):
# =============================================================================
#     Rotates an image around a user-defined pivot point.
#     Note: Currently, pivot[0] = x coord and pivot[1] = y coord while image shape is in row column order. This can be swapped if desired.
#     Copied in from https://stackoverflow.com/questions/25458442/rotate-a-2d-image-around-specified-origin-in-python
# =============================================================================
    padX = [img.shape[1] - pivot[0], pivot[0]]
    padY = [img.shape[0] - pivot[1], pivot[1]]
    imgP = np.pad(img, [padY, padX], 'constant')
    imgR = rotate(imgP, angle, reshape=False)
    return imgR[padY[0] : -padY[1], padX[0] : -padX[1]]

def preprocess_high_res():
# =============================================================================
#     For each image ID specified, saves the image to "High_Res_Input_Images_Processed"
#     Also applies the following:
#         - crops the image
#         - centers cell
#     Note: saves channels separately to preserve bit depth
#     Note 2: need to edit "i" when processing images from different files to prevent overwriting
# =============================================================================
    for i in range(1,file.num_high_res + 1):
        if i == 8: # edit this if removed points from point list manually before acquisition, or somehow missing points
            continue
        else:
            print("Processing image %d" % i)
            DAPI = get_high_res_image(file.hr_DAPI % i)
            atubulin = get_high_res_image(file.hr_atubulin % i)
            thresh = threshold_otsu(DAPI)
            DAPI_closed = closing(DAPI > thresh) # get rid of speckles 
            labeled_DAPI = label(DAPI_closed) # label the remaining objects
#            print("labeled_DAPI:", labeled_DAPI) # Outputs a list of lists
            
            num_nucleus = 0 # initialize the numer of nuclei found per image as zero
            
            properties = regionprops(labeled_DAPI, DAPI)
                    
            for j in range(len(properties)): # identifies nuclei
                if properties[j].area > 2500 and properties[j].area < 30000: # initially 2500, 10000
                    num_nucleus += 1
                    ID = j
                    
                   
            if (num_nucleus == 1): # removes multiple cells / telophase
                global num_cells

                intensity = properties[ID].mean_intensity
                area = properties[ID].area
                blob = properties[ID].intensity_image
                std = np.nanstd(np.where(np.isclose(blob,0), np.nan, blob))
                g_value = (intensity) * (std)**3 / (area)
                
                x_center = properties[ID].centroid[0]
#                print("x_center type:",type(x_center))
                y_center = properties[ID].centroid[1]
                
                x_min = int(int(x_center) - c_w/2)
                x_max = int(int(x_center) + c_w/2)
                y_min = int(int(y_center) - c_w/2)
                y_max = int(int(y_center) + c_w/2)
                    
                if (g_value > 70000000): # We've found a mitotic cell. Initially set to 75000000

                    # Want to rotate image in 1 degree increments through image and identify orientation where line integral has highest intensity
                    intensity_list = [] # list of total intensity along line for each rotation
                    for i in range(360):
                        rotated_image = rotateImage(DAPI, i, (round(x_center),round(y_center))) # define pivot point at the centroid
                        # Sum pixel intensity along vertical line from centroid.
                        tot_intensity = 0
                        for y in range(c_w):
                            tot_intensity += rotated_image[round(y_center)+y][round(x_center)] # add each pixel value to total intensity
                        intensity_list.append(tot_intensity)
                    max_rotation = np.argmax(intensity_list) # find the index/rotation of highest intensity (if tie exists, the return first index)
                    # Before cropping, align image channels so that axis with brightest DAPI points up.
                    aligned_DAPI = rotateImage(DAPI, max_rotation, (round(x_center),round(y_center)))
                    aligned_atubulin = rotateImage(atubulin, max_rotation, (round(x_center),round(y_center)))
                    
                    # Save cropped image of aligned mitotic cell to folder.
#                    num_cells += 1
#                    file_num = num_cells + 96
#                    parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
#                    DAPI_path = os.path.join(parent, "High_Res_Input_Images_Processed", "test_lc1", "DAPI_%d.tif" % file_num)
#                    atubulin_path = os.path.join(parent, "High_Res_Input_Images_Processed","test_lc1", "atubulin_%d.tif" % file_num)
##                    print("Number calls:",num_calls)
                    
                    parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
                    DAPI_path = os.path.join(parent, "High_Res_Input_Images_Processed", "DAPI_%d.tif" % i)
                    atubulin_path = os.path.join(parent, "High_Res_Input_Images_Processed", "atubulin_%d.tif" % i)
                    
                    DAPI_cropped = aligned_DAPI[x_min:x_max, y_min:y_max]
                    atubulin_cropped = aligned_atubulin[x_min:x_max, y_min:y_max]
    
                    io.imsave(DAPI_path, DAPI_cropped) 
                    io.imsave(atubulin_path, atubulin_cropped) 
                    
    # Continue by flipping the cropped images to get the area of highest pixel intensity in a corner.



        
        

        
        
        
        
        
        
        
        
        
        
        

import os 
from skimage import io
from image_display import display_image
import file
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import closing
from skimage.measure import label, regionprops
import numpy as np

c_w = 200


def get_high_res_image(name):
# =============================================================================
#     Gets a high res (60X) image from the "High_Res_Input_Images" folder
#     and returns it as a matrix
# =============================================================================

    parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    image_path = os.path.join(parent, "High_Res_Input_Images", name)
    image = io.imread(image_path)        
    
    return image

def preprocess_high_res():
# =============================================================================
#     For each image ID specified, saves the image to "High_Res_Input_Images_Processed"
#     Also applies the following:
#         - crops the image
#         - centers cell
#     Note: saves channels separately to preserve bit depth
# =============================================================================
    
    for i in range(1,file.num_high_res + 1):
        print("Processing image %d" % i)
        DAPI = get_high_res_image(file.hr_DAPI % i)
        atubulin = get_high_res_image(file.hr_atubulin % i)
        thresh = threshold_otsu(DAPI)
        DAPI_closed = closing(DAPI > thresh)
        labeled_DAPI = label(DAPI_closed)
        
        num_nucleus = 0
        
        properties = regionprops(labeled_DAPI, DAPI)
                
        for j in range(len(properties)): # identifies nuclei
            if properties[j].area > 2500 and properties[j].area < 10000:
                num_nucleus += 1
                ID = j
                
                
        if (num_nucleus == 1): # removes multiple cells / telophase
            intensity = properties[ID].mean_intensity
            area = properties[ID].area
            blob = properties[ID].intensity_image
            std = np.nanstd(np.where(np.isclose(blob,0), np.nan, blob))
            g_value = (intensity) * (std)**3 / (area)
            
            x_center = properties[ID].centroid[0]
            y_center = properties[ID].centroid[1]
            
            x_min = int(int(x_center) - c_w/2)
            x_max = int(int(x_center) + c_w/2)
            y_min = int(int(y_center) - c_w/2)
            y_max = int(int(y_center) + c_w/2)
                
            if (g_value > 75000000):
                parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
                DAPI_path = os.path.join(parent, "High_Res_Input_Images_Processed", "DAPI_%d.tif" % i)
                atubulin_path = os.path.join(parent, "High_Res_Input_Images_Processed", "atubulin_%d.tif" % i)
                
                DAPI_cropped = DAPI[x_min:x_max, y_min:y_max]
                atubulin_cropped = atubulin[x_min:x_max, y_min:y_max]

                io.imsave(DAPI_path, DAPI_cropped) 
                io.imsave(atubulin_path, atubulin_cropped) 



        
        

        
        
        
        
        
        
        
        
        
        
        

import display

import os
from skimage import io
import matplotlib.pyplot as plt
from skimage.feature import blob_log, blob_doh
from math import sqrt
from skimage import filters
from skimage.measure import label, regionprops
from skimage.exposure import rescale_intensity
import numpy as np

gaussian_std = 10

high_res_cells = {}
high_res_patterns = {}

high_res_valid = {}

image_size = 1024

c_w = 300

def export_valid_images():
    for i in high_res_valid.keys():
        file_path = os.path.join(os.getcwd(), 'Cropped_High_Res_Images', 'Image_%d.tif' % i)
        
        
        
        atubulin = get_high_res_image("1_192_2_%.4d_R3D_D3D_PRJ_w523.tif" % i)
        pH3 = get_high_res_image("1_192_2_%.4d_R3D_D3D_PRJ_w632.tif" % i)
        pattern = get_high_res_image("1_192_2_%.4d_R3D_D3D_PRJ_w676.tif" % i)

        atubulin_rescale = rescale_intensity(atubulin)
        pH3_rescale = rescale_intensity(pH3)
        pattern_rescale = rescale_intensity(pattern, in_range = (0, 10000))

        atubulin_8 = (atubulin_rescale/256).astype('uint8')
        pH3_8 = (pH3_rescale/256).astype('uint8')
        pattern_8 = (pattern_rescale/256).astype('uint8')
        
    
        uncropped = np.dstack((pattern_8, atubulin_8, pH3_8))
        uncropped[:, :, 0] = 0 # comment out to add pattern channel
            
        x_val = int(high_res_cells[i][1])
        y_val = int(high_res_cells[i][0])
        
        x_min = int(x_val - c_w/2)
        x_max = int(x_val + c_w/2)
        y_min = int(y_val - c_w/2)
        y_max = int(y_val + c_w/2)
        
        if (0 <= x_min <= 1023 and 0 <= x_max <= 1023 and 0 <= y_min <= 1023 and 0 <= y_max <= 1023):
        
            cropped = uncropped[x_min:x_max, y_min:y_max, :]
            io.imsave(file_path, cropped)       
    

def get_valid_images():
    for key in high_res_cells.keys():
        if key in high_res_patterns.keys():
            cell_x = high_res_cells[key][0]
            cell_y = high_res_cells[key][1]
            
            pattern_x = high_res_patterns[key][0]
            pattern_y = high_res_patterns[key][1]
            
            dist = sqrt((cell_x - pattern_x)**2 + (cell_y - pattern_y)**2)
            
            if (dist < 50):
                high_res_valid[key] = dist
            



def get_high_res_image(name):
    return io.imread(os.path.join(os.getcwd(), 'High_Res_Images', name))

class High_Res_Image:
    def __init__(self, image_file, image_id):
        self.image_file = image_file
        self.image_id = image_id

    def display_image(self):
        fig, ax = plt.subplots(1, figsize = (10,10)) 
        ax.imshow(self.image_file, cmap='gray', interpolation='nearest')
        ax.set_aspect('equal')
        plt.show()   
        
    def apply_image_otsu_pattern(self):
        gaus = filters.gaussian(self.image_file,gaussian_std)
        im = filters.threshold_otsu(gaus)
        fil = im < (gaus * 0.8)
        label_image = label(fil)      
        
        prop = regionprops(label_image)
        
        temp = []
        
        for i in range(len(prop)):
            
                                    
            if (50000 < prop[i].area < 100000):
                centroid_x = prop[i].centroid[1]
                centroid_y = prop[i].centroid[0]
                
                temp.append([i, centroid_x, centroid_y])
                
                
                        
        if (len(temp) != 0):
            curr_id = 0

            min_dist = float("inf")
                
            for i in range(len(temp)):
                dist = sqrt((image_size/2 - temp[i][1])**2 + (image_size/2 - temp[i][2])**2)
                if (dist < min_dist):
                    min_dist = dist
                    curr_id = i
                    
            high_res_patterns[self.image_id] = [temp[curr_id][1], temp[curr_id][2]]
                    
#        label_image = label(fil)      
#        fig, ax = plt.subplots(1, figsize = (10,10)) 
#        ax.imshow(label_image, cmap='gray', interpolation='nearest')
#        ax.set_aspect('equal')
#        plt.show() 
    
        
    def apply_image_otsu_cell(self):
        
        gaus = filters.gaussian(self.image_file,gaussian_std)
        im = filters.threshold_otsu(gaus)
        fil = im < gaus
        label_image = label(fil)      
        
        prop = regionprops(label_image)
        
        temp = []
        
        for i in range(len(prop)):
            if (10000 < prop[i].area < 30000):
                centroid_x = prop[i].centroid[1]
                centroid_y = prop[i].centroid[0]
                
                temp.append([i, centroid_x, centroid_y])
                
                        
        if (len(temp) != 0):
            curr_id = 0

            min_dist = float("inf")
                
            for i in range(len(temp)):
                dist = sqrt((image_size/2 - temp[i][1])**2 + (image_size/2 - temp[i][2])**2)
                if (dist < min_dist):
                    min_dist = dist
                    curr_id = i
                    
                    
            high_res_cells[self.image_id] = [temp[curr_id][1], temp[curr_id][2]]   
                    
    
        
#        label_image = label(fil)      
#        fig, ax = plt.subplots(1, figsize = (10,10)) 
#        ax.imshow(label_image, cmap='gray', interpolation='nearest')
#        ax.set_aspect('equal')
#        plt.show() 
        
        
    def segment_image(self):
     
        self.objects = blob_log(self.image_file, min_sigma = 500)
        
        
        
        
        
        
        
        
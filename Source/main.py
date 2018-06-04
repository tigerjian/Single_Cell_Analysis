import low_res_analysis
from low_res_analysis import get_low_res_DAPI_image, get_low_res_pattern_image
from image_display import display_image
import file
import DV_calibration
import numpy as np
import high_res_analysis
from skimage.filters import threshold_otsu, gaussian
import linear_LCM_calibration
import non_linear_LCM_calibration
import non_linear_LCM_calibration_V2
import leastsq_LCM_calibration

def analyze_low_res():
    '''
    Analyzes the low res images and produces a point list for cells of interest
    
    '''
    image_coord = file.get_low_res_coord()        
    
    for i in range(1, file.num_low_res + 1):   
        print("Analyzing image %d" % i)
    
        DAPI_img = get_low_res_DAPI_image(file.DAPI_file % i)            
        pattern_img = get_low_res_pattern_image(file.pattern_file % i)            
    
        image = low_res_analysis.Low_Res_Image(DAPI_img, pattern_img,i) 
        
        image.g_method_DAPI()
        image.detect_pattern()
        image.detect_objects()
        
        image.transform_coord(image_coord)
    
    low_res_analysis.save_points()
                
    print("%d points of interest found" % len(low_res_analysis.cell_coord))
        
    DV_calibration.run_calibration(low_res_analysis.cell_coord)      
    DV_calibration.generate_coord(low_res_analysis.cell_coord)
    
def analyze_high_res():
    for i in range(600,700):
        print("Image %d" % i)
        DAPI = high_res_analysis.get_high_res_image(file.hr_DAPI % i)
        atubulin = high_res_analysis.get_high_res_image(file.hr_atubulin % i)
        pattern = high_res_analysis.get_high_res_image(file.hr_pattern % i)
        
        display_image(atubulin)
        
def LCM_calibration():
    leastsq_LCM_calibration.get_points()
    leastsq_LCM_calibration.run_calibration_LCM(leastsq_LCM_calibration.pre_LCM_pts)
    leastsq_LCM_calibration.generate_coord_LCM()

if __name__ == "__main__":
    LCM_calibration()
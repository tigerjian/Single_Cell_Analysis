import low_res_analysis
from low_res_analysis import get_low_res_DAPI_image, get_low_res_pattern_image
from image_display import display_image
import file
import DV_calibration
import numpy as np


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
                
    print("%d points of interest found" % len(low_res_analysis.cell_coord))
        
    DV_calibration.run_calibration(low_res_analysis.cell_coord)      
    DV_calibration.generate_coord(low_res_analysis.cell_coord)
    
    

if __name__ == "__main__":
    analyze_low_res()
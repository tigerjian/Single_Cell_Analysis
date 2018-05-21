import low_res_analysis
from low_res_analysis import get_low_res_DAPI_image, get_low_res_a_tubulin_image, get_low_res_pattern_image
from image_display import display_image
import file


def analyze_low_res():
    '''
    Analyzes the low res images and produces a point list for cells of interest
    
    '''
    
    
    image_coord = file.get_low_res_coord()
        
    for i in range(1,10): #287, 286 total
        print("Analyzing image %d" % i)
    
        DAPI_img = get_low_res_DAPI_image(file.DAPI_file % i)            
        a_tubulin_img = get_low_res_a_tubulin_image("DAPI_atubulin_pattern_1_R3D_PRJ_w632_t%.3d.tif" % i)            
        pattern_img = get_low_res_pattern_image("DAPI_atubulin_pattern_1_R3D_PRJ_w676_t%.3d.tif" % i)            
    
        image = low_res_analysis.Low_Res_Image(DAPI_img, a_tubulin_img, pattern_img,i)
        
        image.detect_DAPI()
        image.detect_a_tubulin()
        image.detect_pattern()
        image.detect_objects()
        
        image.transform_coord(image_coord)
    
    

if __name__ == "__main__":
    analyze_low_res()
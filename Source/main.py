import low_res_analysis
from low_res_analysis import get_low_res_DAPI_image, get_low_res_a_tubulin_image, get_low_res_pattern_image
from image_display import display_image


def analyze_low_res():
    '''
    Analyzes the low res images 
    
    '''
    
    i = 120
    
    DAPI_img = get_low_res_DAPI_image("DAPI_atubulin_pattern_1_R3D_PRJ_w435_t%.3d.tif" % i)            
    a_tubulin_img = get_low_res_a_tubulin_image("DAPI_atubulin_pattern_1_R3D_PRJ_w632_t%.3d.tif" % i)            
    pattern_img = get_low_res_pattern_image("DAPI_atubulin_pattern_1_R3D_PRJ_w676_t%.3d.tif" % i)            

    image = low_res_analysis.Low_Res_Image(DAPI_img, a_tubulin_img, pattern_img)
    image.detect_a_tubulin()
    
    display_image(a_tubulin_img)
    
    

if __name__ == "__main__":
    analyze_low_res()
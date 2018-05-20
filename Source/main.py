import low_res_analysis
from low_res_analysis import get_low_res_DAPI_image, get_low_res_a_tubulin_image, get_low_res_pattern_image
from image_display import display_image


def analyze_low_res():
    '''
    Analyzes the low res images 
    
    '''
    image_mat = get_low_res_pattern_image("DAPI_atubulin_pattern_1_R3D_PRJ_w676_t001.tif")        
    
    display_image(image_mat)
    
    image = low_res_analysis.Low_Res_Image(image_mat)
    
    

if __name__ == "__main__":
    analyze_low_res()
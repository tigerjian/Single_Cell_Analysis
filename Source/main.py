from low_res_analysis import get_low_res_DAPI_image


def analyze_low_res():
    '''
    Analyzes the low res images 
    
    '''
    
    image = get_low_res_DAPI_image("DAPI_atubulin_pattern_1_R3D_PRJ_w435_t001.tif")    
    
    

if __name__ == "__main__":
    analyze_low_res()
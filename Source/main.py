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
import high_res_preprocessing
import SIFT
import os
import tSNE
import DBSCAN
import PCA_decomp
import ICA_decomp
import DEC
import DEC_w_VAE


def analyze_low_res():
    '''
    Analyzes the low res images and produces a point list for cells of interest
    
    '''
    image_coord = file.get_low_res_coord()        
    
    for i in range(1, file.num_low_res + 1): 
#    for i in np.arange(1, file.num_low_res + 1, 500):
        print("Analyzing image %d" % i)
    
        DAPI_img = get_low_res_DAPI_image(file.DAPI_file % i)            
    
        image = low_res_analysis.Low_Res_Image(DAPI_img,i) 
        
        image.g_method_DAPI()
        image.detect_DAPI()
        image.detect_objects()
        
        image.transform_coord(image_coord)
    
    low_res_analysis.save_points()
                
    print("%d points of interest found" % len(low_res_analysis.cell_coord))
        
    DV_calibration.run_calibration(low_res_analysis.cell_coord)      
    DV_calibration.generate_coord(low_res_analysis.cell_coord)
    
def analyze_high_res():
    for i in range(1,10):
        print("Image %d" % i)
        DAPI = high_res_analysis.get_high_res_image(file.hr_DAPI % i)
        atubulin = high_res_analysis.get_high_res_image(file.hr_atubulin % i)
        pattern = high_res_analysis.get_high_res_image(file.hr_pattern % i)
        
        display_image(atubulin)
        
def LCM_calibration():
# =============================================================================
#     performs nonlinear calibration for the LCM scope using leastsq
# =============================================================================
    leastsq_LCM_calibration.get_points()
    leastsq_LCM_calibration.run_calibration_LCM(leastsq_LCM_calibration.pre_LCM_pts)
    leastsq_LCM_calibration.generate_coord_LCM()
    
def generate_descriptors():
# =============================================================================
#     Generates SIFT descriptors for every image/cell and applies KMeans
# =============================================================================
    for i in range(1,file.num_high_res + 1):
        parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        image_path = os.path.join(parent, "High_Res_Input_Images_Processed", "DAPI_%d.tif" % i)
        if (os.path.isfile(image_path)):
            print("Generating SIFT descriptors for image %d" % i)
            DAPI_image = SIFT.get_high_res_image("DAPI_%d.tif" % i)
            atubulin_image = SIFT.get_high_res_image("atubulin_%d.tif" % i)
            cell = SIFT.SIFT_image(DAPI_image, atubulin_image, i)
            cell.find_DAPI_desc()

            
    #DBSCAN.apply_DBSCAN(SIFT.desc_KMeans())
#    SIFT.desc_KMeans()
    for i in range(10):
        tSNE.apply_tSNE(SIFT.desc_KMeans()) # if want to run tSNE after SIFT instead of DBSCAN
        
def generate_PCA_features():
    image_mat = []

    for i in range(1,file.num_high_res + 1):
        parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        image_path = os.path.join(parent, "High_Res_Input_Images_Processed", "DAPI_%d.tif" % i)
        if (os.path.isfile(image_path)):
            flattened_image = PCA_decomp.get_high_res_image(image_path).flatten()
            image_mat.append(flattened_image)
            
    comp_mat = PCA_decomp.apply_PCA_decomp(image_mat)
    tSNE.apply_tSNE(comp_mat)
        

if __name__ == "__main__":        
    DEC.run_DEC()    
#    high_res_preprocessing.preprocess_high_res()
#    LCM_calibration()
    
    
    
    
    
    
    
    
    
    
    
    

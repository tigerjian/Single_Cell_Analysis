from sklearn.decomposition import PCA
import os
from skimage import io
from skimage.exposure import rescale_intensity
import numpy as np
from image_display import display_image

def get_high_res_image(name):
# =============================================================================
#     Gets a a high res preprocessed image from the "High_Res_Input_Images_Processed" folder
# =============================================================================

    parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    image_path = os.path.join(parent, "High_Res_Input_Images_Processed", name)
    image = io.imread(image_path)   
    image = (image/256).astype("uint8")  
    image = rescale_intensity(image)
    
    return image

def apply_PCA_decomp(image_mat):
    pca = PCA(
            n_components = 50
            )
    
    comp_mat = pca.fit_transform(image_mat)
    
#    for i in range(100): # displaying the eigencells
#        display_image(pca.components_[i].reshape(225,225))
#    
#    inv_mat = pca.inverse_transform(comp_mat)
#    
#    for i in range(10): # displaying the eigencells
#        display_image(inv_mat[i].reshape(225,225))
    
    return comp_mat
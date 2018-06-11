from sklearn.decomposition import FastICA

import numpy as np
from image_display import display_image


def apply_ICA_decomp(image_mat):
    ica = FastICA(
            n_components = 50
            )
    
    comp_mat = ica.fit_transform(image_mat)
    
#    for i in range(50): # displaying the eigencells
#        display_image(ica.components_[i].reshape(225,225))
#    
#    inv_mat = ica.inverse_transform(comp_mat)
#    
#    for i in range(10): # displaying the eigencells
#        display_image(inv_mat[i].reshape(225,225))
    
    return comp_mat
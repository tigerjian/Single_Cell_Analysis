from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def apply_tSNE(hist_mat):
# =============================================================================
#     This functions applies tSNE to hist_mat
# =============================================================================
    hist_mat_embedded = TSNE(
            n_components = 2,
            perplexity = 5
            ).fit_transform(hist_mat)
    
    x = hist_mat_embedded[:,1]
    y = hist_mat_embedded[:,0]
    
    plt.scatter(x,y)
    plt.show()



    
    
    
    
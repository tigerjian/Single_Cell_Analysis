from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
import file

def apply_tSNE(hist_mat):
# =============================================================================
#     This functions applies tSNE to hist_mat
# =============================================================================
    
    hist_vecs = np.asarray(hist_mat)
    
    hist_mat_embedded = TSNE(
            n_components = 2,
            perplexity = 3
            ).fit_transform(hist_vecs)
    
    hist_mat_clusters = DBSCAN(
            eps = 5
        ).fit_predict(hist_mat_embedded)
    
    for i in range(1, file.num_high_res + 1):
        print("Class for Image # %d: %d" % (i, hist_mat_clusters[i - 1]))
        
    x = hist_mat_embedded[:,1]
    y = hist_mat_embedded[:,0]
    
    plt.figure(figsize = (10,10))
    plt.axis('equal')
    plt.scatter(x,y)
    plt.show()



    
    
    
    
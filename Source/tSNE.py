from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib  import cm
import matplotlib.colors
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

import file

def apply_tSNE(hist_mat, n, perp, labels, do_kmeans):
# =============================================================================
#     This functions applies tSNE to hist_mat
# =============================================================================
    
    hist_vecs = np.asarray(hist_mat)
    
    hist_mat_embedded = TSNE(
            n_components = 2,
            perplexity = perp
            ).fit_transform(hist_vecs)
#    print("hist_mat_embedded:",hist_mat_embedded)
    
#    # Run DBSCAN on t-SNE output
#    hist_mat_clusters = DBSCAN(
#            eps = 7
#        ).fit_predict(hist_mat_embedded)
#    print("hist_mat_clusters:", hist_mat_clusters)
    
    # Run k-means clustering on t-SNE output
    hist_mat_clusters = KMeans(
            n_clusters = n,
            init = 'k-means++').fit_predict(hist_mat_embedded)
        
#    for i in range(1, file.num_high_res + 1):
#        print("Class for Image # %d: %d" % (i, hist_mat_clusters[i - 1]))
    
    # Want to color code, but we're plotting the tSNE output.
    x = hist_mat_embedded[:,1]
    y = hist_mat_embedded[:,0]
    
    if (do_kmeans == True):
        cluster_labels = hist_mat_clusters
    else:
        cluster_labels = labels
        
    max_label = max(cluster_labels)
    print("max_label:",max_label)
    if max_label != -1:
        num_clusters = max_label + 1 # cluster labels go from -1 to max_label incrementally
    else:
        num_clusters = 1
    
    plt.figure(figsize = (10,10))
    plt.axis('equal')
    # scatter with colormap mapping to the cluster labels from DBSCAN or k-means (returned as array)
    color_map = plt.get_cmap("viridis")
    if max_label != -1:
        norm = matplotlib.colors.BoundaryNorm(np.arange(0,num_clusters + 1,1), color_map.N)
        plt.scatter(x,y,s=25,c=cluster_labels, marker = 'o', cmap = color_map, norm = norm );
        plt.colorbar()
    plt.scatter(x,y,s=25,c=cluster_labels, marker = 'o', cmap = color_map );
    plt.show()
        



    
    
    
    
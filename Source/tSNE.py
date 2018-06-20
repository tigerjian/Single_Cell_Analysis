from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib  import cm
import matplotlib.colors
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

import file
import image_display


x =[]
y = []

def apply_tSNE(hist_mat, n, perp, labels, do_kmeans):
# =============================================================================
#     This functions applies tSNE to hist_mat
# =============================================================================
    global x, y
    
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
        

    
    ## Display interactive, color-coded t-SNE plot    
    # I don't know where the image ID labels come from. Manually:
    test_ID = [i for i in range(126) if i != 43]
    
    # Make a dictionary where keys are cluster assignments and values are the values
    # Doing this to verify that my interactive plot displays the correct images.
    #  Result: the images do appear to match their cluster assignment!
    cluster_ID_dict = {}
    for key,value in zip(cluster_labels,test_ID):
        if key not in cluster_ID_dict:
            cluster_ID_dict[key]=[value]
        else:
            cluster_ID_dict[key].append(value)
    print(cluster_ID_dict)
    
    image_display.interactive_plot(test_ID, cluster_labels, x, y)   

      
#    ## Display non-interactive color-coded t-SNE plot
#    image_display.display_color_coded_plot(x, y, cluster_labels, 'viridis', (10,10)) # for t-SNE plot

    


    
    
    
    
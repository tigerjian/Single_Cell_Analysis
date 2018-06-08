from sklearn.cluster import DBSCAN


def apply_DBSCAN(hist_mat):
# =============================================================================
#     This function applies the DBSCAN algorithm to the input hist_mat
# =============================================================================
    hist_mat_clusters = DBSCAN(
            ).fit_predict(hist_mat)
    
    print(hist_mat_clusters)
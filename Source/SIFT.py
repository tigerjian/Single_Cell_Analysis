import os
from skimage import io
from image_display import display_image
import cv2
import numpy as np
from skimage.exposure import rescale_intensity
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


bag_of_atubulin_desc = [] # stores all the descriptors found using SIFT
bag_of_DAPI_desc = [] # stores all the descriptors found using SIFT


hist_dic = {}

k = 10 # number of clusters for KMeans

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
    
def desc_KMeans():
    kmeans = KMeans(n_clusters = k)
    desc = np.asarray(bag_of_DAPI_desc)[:,:-1]
    
    
    
    
#    hist_mat_embedded = TSNE(
#            n_components = 2,
#            perplexity = 25
#            ).fit_transform(desc)
#    
#    x = hist_mat_embedded[:,1]
#    y = hist_mat_embedded[:,0]
#    
#    plt.figure(figsize = (15,15))
#    plt.axis('equal')
#    plt.scatter(x,y)
#    plt.show()
    
    
    
    cluster_ids = kmeans.fit_predict(desc)
    image_ids = np.asarray(bag_of_DAPI_desc)[:,-1:]
        
    for image_id in np.unique(image_ids):
        hist_dic[int(image_id)] = np.zeros(k)
            
    for i in range(len(cluster_ids)):
        cluster_id = cluster_ids[i] # the cluster ID of the descriptor
        image_id = int(image_ids[i])
        
        hist_dic[image_id][cluster_id] += 1
        
    hist_mat = []
    
    for key in hist_dic.keys():
        hist_vec = hist_dic[key]
        hist_vec = np.append(hist_vec, key)
        hist_mat.append(hist_vec)
                
    return hist_mat
        
class SIFT_image:
# =============================================================================
#     Each object represents an image whose descriptors are to be found
# =============================================================================
    def __init__(self, DAPI, atubulin, image_id):
        self.DAPI = DAPI
        self.atubulin = atubulin
        self.image_id = image_id
        
    def find_DAPI_desc(self):    
        sift = cv2.xfeatures2d.SIFT_create(
                #edgeThreshold = 10,
                nfeatures = 50,
                nOctaveLayers = 50,
                contrastThreshold = 0.005,
                sigma = 3
                )
        
        kp, desc = sift.detectAndCompute(self.DAPI,None)
        
        
        for i in range(len(desc)):
            if (kp[i].size > 25):
                d = np.append(desc[i], self.image_id)     
                bag_of_DAPI_desc.append(d)
            
        
#        img = cv2.drawKeypoints(self.DAPI, kp, self.DAPI.copy(), flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#        display_image(img)
        
    def find_atubulin_desc(self):    
        sift = cv2.xfeatures2d.SIFT_create(
                #edgeThreshold = 10,
                nfeatures = 100,
                nOctaveLayers = 250,
                contrastThreshold = 0.005,
                sigma = 3
                )
        
        kp, desc = sift.detectAndCompute(self.atubulin, None)
        
        for d in desc:
            d = np.append(d, self.image_id)     
            bag_of_DAPI_desc.append(d) # temporary test
                    
#        img = cv2.drawKeypoints(self.atubulin, kp, self.atubulin.copy(), flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#        display_image(img)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    




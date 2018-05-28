from skimage import io
import os
import matplotlib.pyplot as plt
from skimage import filters
from skimage.measure import label, regionprops
import cv2
from skimage.exposure import rescale_intensity
import numpy as np
from image_display import display_image
from math import sqrt
from skimage.filters import threshold_otsu, gaussian
from skimage.measure import label, regionprops
from skimage.morphology import closing, opening
from skimage.segmentation import clear_border
import DV_calibration

gaussian_std = 5
image_size = 2048
pixel_size = 0.32557

cell_coord = []

DAPI_dist = 30

total = 0

def save_points():
    parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    path = os.path.join(parent, "Point_Lists", "pre_LCM")
        
    f = open(path, "w")
        
    for i in range(len(cell_coord)):
        f.write(str(cell_coord[i][0]) + " ")
        f.write(str(cell_coord[i][1]) + " ")
        f.write(str(cell_coord[i][2]) + " ")
        f.write(str(cell_coord[i][3]) + " ")
        f.write(str(cell_coord[i][4]) + "\n")
    f.close()

def get_low_res_DAPI_image(name):
    '''
    Returns the DAPI image in the Low_Res_Input_Images folder as a 2d array
    - converts to 8 bit from 16 bit
    - rescales intensity
    
    '''
    parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    image_path = os.path.join(parent, "Low_Res_Input_Images_20x", "DAPI", name)
    
    image = io.imread(image_path)        
    image = (image/20).astype("uint8")
    #image = rescale_intensity(image)
    
    return image

def get_low_res_pattern_image(name):
    '''
    Returns the pattern image in the Low_Res_Input_Images folder as a 2d array
    
    '''
    parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    image_path = os.path.join(parent, "Low_Res_Input_Images_20x", "Pattern", name)
    
    image = io.imread(image_path)
    image = (image/256).astype("uint8")
    
    mean = np.mean(image)
    std = np.std(image)
    
    image = np.clip(image, 0, ((mean + std) * 3).astype("uint8"))
    image = rescale_intensity(image)
    
    return image

def euc_dist(x1, y1, x2, y2):
    return sqrt((x1 - x2)**2 + (y1 - y2)**2)

class Low_Res_Image:
    
    def __init__(self, DAPI, pattern, image_id):
        self.image_id = image_id
        
        self.DAPI = DAPI  
        self.pattern = pattern  
        
        self.DAPI_pts = []
        self.a_tubulin_pts = []
        self.pattern_pts = []
        self.g_DAPI_pts = []
        
        self.objects = []
        
    def detect_DAPI(self):
        params = cv2.SimpleBlobDetector_Params()
        
        # detector parameters
        params.blobColor = 255;
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        params.filterByArea = False
        params.minThreshold = 100

        detector = cv2.SimpleBlobDetector_create(params)
        
        keypoints = detector.detect(self.DAPI)
        
        for i in range(len(keypoints)):
            self.DAPI_pts.append(keypoints[i].pt)
        
        # uncomment to view labeled image
#        im_with_keypoints = cv2.drawKeypoints(self.DAPI, keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)    
#        display_image(im_with_keypoints)
        
    def detect_a_tubulin(self):
        params = cv2.SimpleBlobDetector_Params()
        params.blobColor = 255;
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        params.filterByArea = True
        params.minArea = 25
        params.maxArea = 250
        params.minThreshold = 50

        detector = cv2.SimpleBlobDetector_create(params)
        
        keypoints = detector.detect(self.a_tubulin)
        
        for i in range(len(keypoints)):
            self.a_tubulin_pts.append(keypoints[i].pt)
        
        # uncomment to view labeled image
#        im_with_keypoints = cv2.drawKeypoints(self.a_tubulin, keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)    
#        display_image(im_with_keypoints)
    
    def detect_pattern(self):
        params = cv2.SimpleBlobDetector_Params()
        params.blobColor = 255;
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        params.filterByArea = True
        params.minArea = 2500
        params.maxArea = 30000

        detector = cv2.SimpleBlobDetector_create(params)
        
        keypoints = detector.detect(self.pattern)
        
        for i in range(len(keypoints)):
            self.pattern_pts.append(keypoints[i].pt)
        
        # uncomment to view labeled image
#        im_with_keypoints = cv2.drawKeypoints(self.pattern, keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)    
#        display_image(im_with_keypoints)
        
    def detect_objects(self):
        global total
        
        for pat in self.g_DAPI_pts:
            near_DAPI = False
#            near_a_tubulin = False
            
            for DAPI in self.pattern_pts:
                d = euc_dist(DAPI[0], DAPI[1], pat[0], pat[1])
                if d < DAPI_dist:
                    near_DAPI = True
                    
#            for a_tubulin in self.a_tubulin_pts:
#                d = euc_dist(a_tubulin[0], a_tubulin[1], pat[0], pat[1])
#                if d < a_tubulin_dist:
#                    near_a_tubulin = True
            
#            if near_DAPI and near_a_tubulin:
            if near_DAPI:
                self.objects.append(pat)
        
    def transform_coord(self, image_coord):        
        image_x = float(image_coord[int(self.image_id)][0])
        image_y = float(image_coord[int(self.image_id)][1])
                
        for i in range(len(self.objects)):
            x_val = self.objects[i][0]
            y_val = self.objects[i][1]
            
            x_coord = image_x + (image_size/2 - x_val) * pixel_size
            y_coord = image_y + (y_val - image_size/2) * pixel_size
            cell_coord.append([x_coord, y_coord, x_val, y_val, self.image_id])
            
            
#        fig, ax = plt.subplots(1, figsize = (15,15))
#        # adding labels
#        for i in range(len(self.objects)):
#            c = plt.Circle((self.objects[i][0], self.objects[i][1]), 30, color = 'red', linewidth = 1, fill = False)
#            ax.add_patch(c)
#        
#        
#        ax.imshow(self.DAPI, cmap='gray', interpolation='nearest')
#        ax.set_aspect('equal')
#        plt.axis('off')
#        plt.show()    


    def g_method_DAPI(self):
        thresh = threshold_otsu(self.DAPI)
        DAPI_thresh = self.DAPI > thresh
        DAPI_cleared = clear_border(DAPI_thresh)
        
        DAPI_labeled = label(DAPI_cleared)
        
                
        for region in regionprops(DAPI_labeled, self.DAPI):
            intensity = region.mean_intensity
            area = region.area
            blob = region.intensity_image
            std = np.nanstd(np.where(np.isclose(blob,0), np.nan, blob))
            
            g_value = (intensity) * (std)**3 / (area)
                                                
            if g_value > 500 and area > 100 and area < 3000:
                x = region.centroid[0]
                y = region.centroid[1]
                
                self.g_DAPI_pts.append([y, x])
                
            
#        fig, ax = plt.subplots(1, figsize = (30,30))
        
                
#        # adding labels
#        for i in range(len(self.g_DAPI_pts)):
#            c = plt.Circle((self.g_DAPI_pts[i][0], self.g_DAPI_pts[i][1]), 30, color = 'red', linewidth = 1, fill = False)
#            ax.add_patch(c)
#        
#        
#        ax.imshow(self.DAPI, cmap='gray', interpolation='nearest')
#        ax.set_aspect('equal')
#        plt.axis('off')
#        plt.show()   
                
                

        

   
        
                
        
        
        
        
        
        
        
        
        
        
        
        
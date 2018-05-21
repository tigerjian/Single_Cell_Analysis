from skimage import io
import os
import matplotlib.pyplot as plt
from skimage import filters
from skimage.measure import label, regionprops
import cv2
from skimage.exposure import rescale_intensity
import numpy as np
from image_display import display_image

gaussian_std = 5
image_size = 2048
pixel_size = 0.64570

def get_low_res_DAPI_image(name):
    '''
    Returns the DAPI image in the Low_Res_Input_Images folder as a 2d array
    - converts to 8 bit from 16 bit
    - rescales intensity
    
    '''
    parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    image_path = os.path.join(parent, "Low_Res_Input_Images", "DAPI", name)
    
    image = io.imread(image_path)
    image = (image/256).astype("uint8")
    image = rescale_intensity(image)
    
    return image

def get_low_res_a_tubulin_image(name):
    '''
    Returns the a_tubulin image in the Low_Res_Input_Images folder as a 2d array
    
    '''
    parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    image_path = os.path.join(parent, "Low_Res_Input_Images", "a_tubulin", name)
    
    image = io.imread(image_path)
    image = (image/256).astype("uint8")
    image = rescale_intensity(image)
    
    return image

def get_low_res_pattern_image(name):
    '''
    Returns the pattern image in the Low_Res_Input_Images folder as a 2d array
    
    '''
    parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    image_path = os.path.join(parent, "Low_Res_Input_Images", "Pattern", name)
    
    image = io.imread(image_path)
    image = (image/256).astype("uint8")
    
    mean = np.mean(image)
    std = np.std(image)
    
    image = np.clip(image, 0, ((mean + std) * 5).astype("uint8"))
    image = rescale_intensity(image)
    
    return image

class Low_Res_Image:
    
    def __init__(self, DAPI, a_tubulin, pattern):
        self.DAPI = DAPI  
        self.a_tubulin = a_tubulin  
        self.pattern = pattern  
        
    def detect_DAPI(self):
        params = cv2.SimpleBlobDetector_Params()
        params.blobColor = 255;
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        params.filterByArea = False

        params.minThreshold = 100

        detector = cv2.SimpleBlobDetector_create(params)
        
        keypoints = detector.detect(self.DAPI)
        
        im_with_keypoints = cv2.drawKeypoints(self.DAPI, keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)    
        
        display_image(im_with_keypoints)
        
    def detect_a_tubulin(self):
        params = cv2.SimpleBlobDetector_Params()
        params.blobColor = 255;
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        params.filterByArea = True
        params.minArea = 25
        params.minThreshold = 50

        detector = cv2.SimpleBlobDetector_create(params)
        
        keypoints = detector.detect(self.a_tubulin)
        
        im_with_keypoints = cv2.drawKeypoints(self.a_tubulin, keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)    
        
        display_image(im_with_keypoints)
    
    def detect_pattern(self):
        params = cv2.SimpleBlobDetector_Params()
        params.blobColor = 255;
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        params.filterByArea = True
        params.minArea = 1000
        params.maxArea = 3000

        detector = cv2.SimpleBlobDetector_create(params)
        
        keypoints = detector.detect(self.pattern)
        
        im_with_keypoints = cv2.drawKeypoints(self.pattern, keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)    
        
        display_image(im_with_keypoints)
        
    def transform_coord(self, image_coord):        
        image_x = float(image_coord[int(self.image_id)][0])
        image_y = float(image_coord[int(self.image_id)][1])
                
        for i in range(self.objects.shape[0]):
            x_val = self.objects[i,1]
            y_val = self.objects[i,0]
            x_coord = image_x + (image_size/2 - x_val) * pixel_size
            y_coord = image_y + (y_val - image_size/2) * pixel_size
            cell_coord.append([x_coord, y_coord, x_val, y_val, self.image_id])


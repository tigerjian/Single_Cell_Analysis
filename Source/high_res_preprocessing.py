import os 
from skimage import io
from image_display import display_image
import file
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import closing
from skimage.measure import label, regionprops
import numpy as np
import scipy.ndimage
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import math


c_w = 250 # set cropped image width
num_cells = 0

def get_high_res_image(name):
# =============================================================================
#     Gets a high res (60X) image from the "High_Res_Input_Images" folder
#     and returns it as a matrix
# =============================================================================
    
    parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    image_path = os.path.join(parent, "High_Res_Input_Images", name)
    image = io.imread(image_path)        
    return image

def rotateImage(img, angle, pivot):
# =============================================================================
#     Rotates an image around a user-defined pivot point.
#     Note: Currently, pivot[0] = x coord and pivot[1] = y coord while image shape is in row column order. This can be swapped if desired.
#     Copied in from https://stackoverflow.com/questions/25458442/rotate-a-2d-image-around-specified-origin-in-python
# =============================================================================
    padX = [int(img.shape[1] - pivot[0]), int(pivot[0])] # don't really like having to add int in here, but it gets rid of the pad_width TypeError
    padY = [int(img.shape[0] - pivot[1]), int(pivot[1])]
    imgP = np.pad(img, [padY, padX], 'constant')
    imgR = rotate(imgP, angle, reshape=False)
    return imgR[padY[0] : -padY[1], padX[0] : -padX[1]]

def intensity_rotated_line(img,angle, pivot):
# =============================================================================
#     Rotates a point around another centerPoint. 
#     Angle is in degrees. 
#     Rotation is counter-clockwise. 
#     Then finds total pixel intensity along line.
#     Output: total pixel intensity across line of length c_w/2
# =============================================================================
    
    # move up by the radius
    global c_w
    x0, y0 = pivot[0], pivot[1]
    x1, y1 = int(x0 + c_w/2), int(y0 + c_w/2)
    
    # then rotate
    angle = math.radians(angle)
    x2 = x0 + math.cos(angle) * (x1 - x0) - math.sin(angle) * (y1 - y0) # Modified from https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python
    y2 = y0 + math.sin(angle) * (x1 - x0) + math.cos(angle) * (y1 - y0)
    
    length = int(np.hypot(x2-x0, y2-y0))
    x, y = np.linspace(x0, x2, length), np.linspace(y0, y2, length)
    
    # Extract the values along the line
    intensities = img[x.astype(np.int), y.astype(np.int)]
    tot_intensity = 0
    for pixel in intensities:
        tot_intensity += pixel
    return tot_intensity


def flip_all_if_brighter(channels, fold_line, axis):
# =============================================================================
#     Note: Assumming DAPI will always be the zeroth index of channels (which should be a list of images)
#     Output: list containing flipped images for all channels
# =============================================================================
#    print("len(channels):", len(channels))
    flipped_images = []
    for j in range(len(channels)):     
        flipped_images.append(channels[j]) # initialize filled_images with the initial images
    ref_image = channels[0] # In our case, this will be DAPI
    if axis == 'vertical':
        # Find intensities of upper and lower half of image in DAPI channel
        intensity_top = np.sum(ref_image[:fold_line])
        intensity_bottom = np.sum(ref_image[fold_line:])
        # Flip all channels so top of DAPI image is brighter
        if intensity_top < intensity_bottom:
            for i in range(len(channels)):
                flipped_images[i] == np.flipud(channels[i])
    else: # elif axis == 'horizontal'
        # Find intensities of the left and right half of image
        intensity_left = np.sum(ref_image[:, :fold_line])
        intensity_right = np.sum(ref_image[:, fold_line:])
        # Flip all channels so left of DAPI image is brighter
        if intensity_left < intensity_right:
            for i in range(len(channels)):
                flipped_images[i] = np.fliplr(channels[i])
    return flipped_images # the images in this list will have been flipped if appropriate.
        

def preprocess_high_res():
# =============================================================================
#     For each image ID specified, saves the image to "High_Res_Input_Images_Processed"
#     Also applies the following:
#         - aligns by assigning the brightest axis on DAPI as "up"
#         - crops the image
#         - centers cell
#         - flips images so that brightest DAPI in upper left (flipped vertically, then horizontally)
#     Note: saves channels separately to preserve bit depth
# =============================================================================
    for i in range(1,file.num_high_res + 1):
        if i == 0: # edit this if removed points from point list manually before acquisition, or somehow missing points
            continue
        else:
            print("Processing image %d" % i)
            DAPI = get_high_res_image(file.hr_DAPI % i)
            atubulin = get_high_res_image(file.hr_atubulin % i)
            thresh = threshold_otsu(DAPI)
            DAPI_closed = closing(DAPI > thresh) # get rid of speckles 
            labeled_DAPI = label(DAPI_closed) # label the remaining objects
            
            num_nucleus = 0 # initialize the numer of nuclei found per image as zero
            
            properties = regionprops(labeled_DAPI, DAPI)
                    
            for j in range(len(properties)): # identifies nuclei
                if properties[j].area > 2500 and properties[j].area < 30000: # initially 2500, 10000
                    num_nucleus += 1
                    ID = j
                    
                   
            if (num_nucleus == 1): # removes multiple cells / telophase
                global num_cells

                intensity = properties[ID].mean_intensity
                area = properties[ID].area
                blob = properties[ID].intensity_image
                std = np.nanstd(np.where(np.isclose(blob,0), np.nan, blob))
                g_value = (intensity) * (std)**3 / (area)
                
                #### I've changed these - swapped compared to Tiger's old code. ####
                y_center = properties[ID].centroid[0]
                x_center = properties[ID].centroid[1]
                
                x_min = int(int(x_center) - c_w/2)
                x_max = int(int(x_center) + c_w/2)
                y_min = int(int(y_center) - c_w/2)
                y_max = int(int(y_center) + c_w/2)
                
#                ## Display labeled centroid and crop box on uncropped DAPI image:
#                fig, ax = plt.subplots(1, figsize = (8,8))
#               
#                # adding labels
#                c = plt.Circle((x_center, y_center), 30, color = 'red', linewidth = 1, fill = False)
#                ax.add_patch(c)
#                r = plt.Rectangle((x_min,y_min), (x_max-x_min),(y_max-y_min), linewidth=1,edgecolor='r',facecolor='none')
#                ax.add_patch(r)
#                # display image
#                ax.imshow(DAPI, cmap='gray', interpolation='nearest')
#                ax.set_aspect('equal')
#                plt.axis('off')
#                plt.show()
                    
                if (g_value > 70000000): # We've found a mitotic cell. Initially set to 75000000
                    
                    # Instead of rotating image, want to rotate line around the original image in 1 degree increments.
                    # Want a function for drawing a line that takes a pivot point, angle, and length as arguments.
                    result = []
                    for k in range(360):
                        result.append(intensity_rotated_line(DAPI, k, (round(x_center),round(y_center))))                    
                    max_rotation = np.argmax(result) # the angle where the line integral is maximal
#                    print("max_rotation:",max_rotation)
#                    print("max value:", result[max_rotation])
                    
#                    # Want to rotate image in 1 degree increments through image and identify orientation where line integral has highest intensity
#                    # Unfortunately, this method takes ~2 min per image.
#                    intensity_list = [] # list of total intensity along line for each rotation
#                    for k in range(360):
#                        rotated_image = rotateImage(DAPI, k, (round(x_center),round(y_center))) # define pivot point at the centroid
##                        print("rotated_image[250,250]:",rotated_image[250,250]) # Returns a single value of type 'numpy.uint16'. Key: can't access rotated_image[0][1] - not a valid index
#                        # Sum pixel intensity along vertical line from centroid.
#                        tot_intensity = 0
#                        for y in range(int(c_w/2)): # sum the intensity along the radius of the line
#                            try:
#                                tot_intensity += rotated_image[int(round(y_center)+y), int(round(x_center))] # add each pixel value to total intensity
#                            except IndexError: # line extends beyond edges of image. Add zeros
#                                tot_intensity += 0
##                                continue
#                        intensity_list.append(tot_intensity)
#                    max_rotation = np.argmax(intensity_list) # find the index/rotation of highest intensity (if tie exists, thens return first index)
                    
                    # Before cropping, align image channels so that axis with brightest DAPI points up.
                    aligned_DAPI = rotateImage(DAPI, max_rotation, (round(x_center),round(y_center))) # Note that rotateImage takes point as (x,y) coordinate
                    aligned_atubulin = rotateImage(atubulin, max_rotation, (round(x_center),round(y_center)))
                    
                    parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
                    DAPI_path = os.path.join(parent, "High_Res_Input_Images_Processed", "DAPI_%d.tif" % i)
                    atubulin_path = os.path.join(parent, "High_Res_Input_Images_Processed", "atubulin_%d.tif" % i)
                    
                    # Note that this is opposite the notation that Tiger used before.
                    DAPI_cropped = aligned_DAPI[y_min:y_max, x_min:x_max] 
                    atubulin_cropped = aligned_atubulin[y_min:y_max, x_min:x_max]
                    
                    # Continue by flipping the cropped images to get the area of highest pixel intensity in a corner.
                    # Flip cropped images vertically so total DAPI intensity is higher in top half
                    y_cent = round(DAPI_cropped.shape[0]/2) # number of pixels in y-dir
                    x_cent = round(DAPI_cropped.shape[1]/2) # number of pixels in x-dir (should be same as y-dir). Don't think my code here is correct.
                    channels_to_flip = [DAPI_cropped, atubulin_cropped]
                    channels_DAPI_brighter_top = flip_all_if_brighter(channels_to_flip, y_cent, 'vertical') # list of all flipped images
                    channels_DAPI_brighter_left = flip_all_if_brighter(channels_DAPI_brighter_top, x_cent, 'horizontal') # list of all flipped images
                
                    # Want to save these aligned, cropped, and brightness-flipped images to a folder for analysis.
                    DAPI_processed = channels_DAPI_brighter_left[0]
                    atubulin_processed = channels_DAPI_brighter_left[1]

                    io.imsave(DAPI_path, DAPI_processed) 
                    io.imsave(atubulin_path, atubulin_processed) 
                    
    
        
        

        
        
        
        
        
        
        
        
        
        
        

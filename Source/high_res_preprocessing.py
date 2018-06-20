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
from scipy.stats import tstd
import matplotlib.pyplot as plt
import math
from math import pi
import cv2
from matplotlib.lines import Line2D
from skimage.exposure import rescale_intensity



c_w = 256 # set cropped image width. The CNN likes factors of 2.
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
#     Rotates an image around a user-defined pivot point. Takes angle in radians.
#     Note: Currently, pivot[0] = x coord and pivot[1] = y coord while image shape is in row column order. This can be swapped if desired.
#     Copied in from https://stackoverflow.com/questions/25458442/rotate-a-2d-image-around-specified-origin-in-python
# =============================================================================
    padX = [int(img.shape[1] - pivot[0]), int(pivot[0])] # convert to int to avoid pad_width TypeError
    padY = [int(img.shape[0] - pivot[1]), int(pivot[1])]
    imgP = np.pad(img, [padY, padX], 'constant')
    imgR = rotate(imgP, math.degrees(angle), reshape=False) # Convert angle since this function takes angle in degrees!
    imgF = imgR[padY[0] : -padY[1], padX[0] : -padX[1]]
    return imgF

def rotate_point(pivot, angle, length):
# =============================================================================
#     Rotate a vertical line of given length around a pivot. Assumes zero degrees is straight up. 
#     Angle given in radians.
#     Returns coordinates of rotated point as [x,y].
# =============================================================================
    # move up by the radius
    x0, y0 = pivot[0], pivot[1] ### had to swap this. Tiger
    x1, y1 = int(x0), int(y0 - length) 

    # then rotate
    x2 = x0 + math.cos(angle) * (x1 - x0) - math.sin(angle) * (y1 - y0) # Modified from https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python
    y2 = y0 + math.sin(angle) * (x1 - x0) + math.cos(angle) * (y1 - y0)
    
    return [x2, y2]

def intensity_rotated_line(img,angle, pivot):
# =============================================================================
#     Rotates a vertical line around a pivot point. 
#     Angle is in radians. 
#     Rotation is counter-clockwise. 
#     Output: average pixel intensity across line of length approx. c_w/2 starting at user-defined pivot point.
# =============================================================================
    
    # move up by the radius
    global c_w
    x0, y0 = pivot[0], pivot[1]

    # then rotate both points around
    x1, y1 = rotate_point(pivot, angle, c_w/2) ### Tiger.

    length = int(np.hypot(x1-x0, y1-y0)) # should be same as int(c_w/2)
    x, y = np.linspace(x0, x1, length), np.linspace(y0, y1, length)
    
    # Extract the values along the line
    intensities = []
    try:
        intensities = img[y.astype(np.int), x.astype(np.int)]
    except IndexError:
        print("Line integral extended beyond image.")
           
    # Return the average intensity along the line.
    avg_intensity = np.mean(intensities)
    
        
#    ## Display line on image for each rotation:
#    fig, ax = plt.subplots(1, figsize = (5,5))
#    line = Line2D([x0,x1], [y0, y1], linewidth = 2, color = 'red')
#    ax.add_line(line)
#    ax.imshow(img, cmap='gray', interpolation='nearest')
#    ax.set_aspect('equal')
#    plt.axis('off')
#    plt.show()
#    print("avg_intensity:",(avg_intensity))
    
    return avg_intensity

def pad_image(img):
# =============================================================================
#     Pads images by c_w on each side. Returns the padded image.
# =============================================================================
    global c_w
    # Make array of zeros, adding padding of size crop width to all sides of the image
    y = img.shape[0] + 2*c_w
    x = img.shape[1] + 2*c_w
    shape = (y,x)
    result = np.zeros(shape)
    
    # Replace part of array with the original image starting at (c_w,c_w):
    result[c_w:img.shape[0]+c_w, c_w:img.shape[1]+c_w] = img
    return result
    
    # Both of the following are fairly slow:
#    padded = np.pad(img, ((c_w,c_w),(c_w,c_w)), mode = 'constant', )
#    padded = cv2.copyMakeBorder(img, c_w, c_w, c_w, c_w, cv2.BORDER_CONSTANT,value=0)
#    return padded


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
        intensity_top = np.mean(ref_image[:fold_line, :])
        intensity_bottom = np.mean(ref_image[fold_line:, :])
        # Flip all channels so top of DAPI image is brighter
        if intensity_top < intensity_bottom:
            for i in range(len(channels)):
                flipped_images[i] = np.flipud(channels[i])
    else: # elif axis == 'horizontal'
        # Find intensities of the left and right half of image
        intensity_left = np.mean(ref_image[:, :fold_line])
        intensity_right = np.mean(ref_image[:, fold_line:])
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
    for i in range(1, file.num_high_res + 1):
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
                if properties[j].area > 3000 and properties[j].area < 50000: # initially 2500, 10000
                    num_nucleus += 1
                    ID = j
                    
                  
            if (num_nucleus == 1): # removes multiple cells / telophase
                global num_cells

                intensity = properties[ID].mean_intensity
                area = properties[ID].area
                blob = properties[ID].intensity_image
                std = np.nanstd(np.where(np.isclose(blob,0), np.nan, blob))
                g_value = (std)**3 / (area)
                
                                                                
                #### I've changed these - swapped compared to Tiger's old code. ####
                y_center = properties[ID].centroid[0]
                x_center = properties[ID].centroid[1]
                
                x_min = int(int(x_center) - c_w/2)
                x_max = int(int(x_center) + c_w/2)
                y_min = int(int(y_center) - c_w/2)
                y_max = int(int(y_center) + c_w/2)

                # Pad only if padding needed:
                if x_min < 0  or y_min < 0 or x_max > DAPI.shape[1] or y_max > DAPI.shape[0]:
                    DAPI = pad_image(DAPI)
                    atubulin = pad_image(atubulin)
                    # Update values for the center and crop box:
                    y_center += c_w
                    x_center += c_w
                    x_min += c_w
                    x_max += c_w
                    y_min += c_w
                    y_max += c_w
                        
                
                if (g_value > 500000): # We've found a mitotic cell. Initially set to 75000000
                    
                    # Instead of rotating image, want to rotate line of around the original image.
                    # Blur the image first with Gaussian filter.
                    DAPI_blurred = gaussian(DAPI, sigma = 3)
                    atubulin_blurred = gaussian(atubulin, sigma = 3)
                    result = []
                    fractions = 360 # how many rotated lines to test. 25 is plenty.
                    for k in range(fractions):
                        result.append(intensity_rotated_line(atubulin_blurred, 2*pi*k/fractions, (round(x_center),round(y_center))))                    
                    index_max_rotation = np.argmax(result) # the index corresponding to the angle where the line integral is maximal
                    # Convert from index to angle
                    max_rotation = (index_max_rotation) * 2*pi / fractions

#                    print("index_max_rotation:",index_max_rotation)
#                    print("max_rotation:",max_rotation)
#                    print("max value:", result[index_max_rotation])
                    
#                    ## Display labeled centroid and crop box on uncropped DAPI image:
#                    fig, ax = plt.subplots(1, figsize = (5,5))
#                   
#                    # adding labels
#                    c = plt.Circle((x_center, y_center), 30, color = 'red', linewidth = 1, fill = False)
#                    r = plt.Rectangle((x_min,y_min), (x_max-x_min),(y_max-y_min), linewidth=1,edgecolor='r',facecolor='none')
#                    l = plt.Rectangle((x_center, y_center), 2, c_w/2, angle = math.degrees(max_rotation)) # Use negative length b/c of rotation dir and coordinates. Takes angle in degrees!
#                    ax.add_patch(c)
#                    ax.add_patch(r)
#                    ax.add_patch(l)
#                    # display image
#                    ax.imshow(DAPI, cmap='gray', interpolation='nearest')
#                    ax.set_aspect('equal')
#                    plt.axis('off')
#                    plt.show()

                     
                    # Before cropping, align image channels so that axis with brightest DAPI points up.
                    # To do this, need to rotate CW by 2*pi-max_rotation (or CCW by max_rotation, but the rotate function goes CCW)
                    aligned_DAPI = rotateImage(DAPI, max_rotation, (round(x_center),round(y_center))) # Note that rotateImage takes point as (x,y) coordinate
                    aligned_atubulin = rotateImage(atubulin, max_rotation, (round(x_center),round(y_center)))
                                      
                    # Save cropped image of aligned mitotic cell to folder.
#                    global num_cells
#                    num_cells += 1
#                    file_num = num_cells + 97
#                    parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
#                    DAPI_path = os.path.join(parent, "High_Res_Input_Images_Processed", "DAPI_%d.tif" % file_num)
#                    atubulin_path = os.path.join(parent, "High_Res_Input_Images_Processed", "atubulin_%d.tif" % file_num)
# #                    print("Number calls:",num_calls)
                    
                    
                    parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
                    DAPI_path = os.path.join(parent, "High_Res_Input_Images_Processed", "DAPI_%d.tif" % i)
                    atubulin_path = os.path.join(parent, "High_Res_Input_Images_Processed", "atubulin_%d.tif" % i)
                    
                    DAPI_cropped = aligned_DAPI[y_min:y_max, x_min:x_max]
                    atubulin_cropped = aligned_atubulin[y_min:y_max, x_min:x_max]
                    
                    DAPI_std = np.std(DAPI_cropped)
                    atubulin_std = np.std(atubulin_cropped)
                    
                    DAPI_mean = np.mean(DAPI_cropped)
                    atubulin_mean = np.mean(atubulin_cropped)
                    
                    DAPI_cropped = np.clip(DAPI_cropped, DAPI_mean + DAPI_std * 1, float("inf")).astype("uint16")
                    atubulin_cropped = np.clip(atubulin_cropped, atubulin_mean + atubulin_std * 1, float("inf")).astype("uint16") 
                    
                    DAPI_cropped = rescale_intensity(DAPI_cropped)
                    atubulin_cropped = rescale_intensity(atubulin_cropped)

                    
#                    print("Aligned and cropped DAPI:")
                    
                    # Continue by flipping the cropped images to get the area of highest pixel intensity in a corner.
                    # Flip cropped images vertically so total DAPI intensity is higher in top half
                    y_cent = round(DAPI_cropped.shape[0]/2) # number of pixels in y-dir
                    x_cent = round(DAPI_cropped.shape[1]/2) # number of pixels in x-dir (should be same as y-dir). Don't think my code here is correct.
                    channels_to_flip = [atubulin_cropped, DAPI_cropped]
                    channels_DAPI_brighter_top = flip_all_if_brighter(channels_to_flip, y_cent, 'vertical') # list of all flipped images
                    channels_DAPI_brighter_left = flip_all_if_brighter(channels_DAPI_brighter_top, x_cent, 'horizontal') # list of all flipped images
                
                    # Want to save these aligned, cropped, and brightness-flipped images to a folder for analysis.
                    DAPI_processed = channels_DAPI_brighter_left[0]
                    atubulin_processed = channels_DAPI_brighter_left[1]
                    
#                    print("Processed DAPI:")
#                    display_image(DAPI_processed)

                    
                    io.imsave(DAPI_path, DAPI_processed) 
                    io.imsave(atubulin_path, atubulin_processed) 
                    
    
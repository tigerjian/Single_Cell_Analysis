import os
from skimage import io

from skimage.morphology import opening, closing


def get_high_res_image(name):
    
    parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    image_path = os.path.join(parent, "High_Res_Input_Images", name)
    image = io.imread(image_path)        
    image = (image/256).astype("uint8")
    #image = rescale_intensity(image)
    
    return image


def High_Res_Image
    
    
    

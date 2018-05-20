from skimage import io
import os
import matplotlib.pyplot as plt
from skimage import filters
from skimage.measure import label, regionprops


gaussian_std = 10
image_size = 2048
pixel_size = 0.64570



def get_low_res_DAPI_image(name):
    '''
    Returns the DAPI image in the Low_Res_Input_Images folder as a 2d array
    
    '''
    parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    image_path = os.path.join(parent, "Low_Res_Input_Images", "DAPI", name)
    
    return io.imread(image_path)

def get_low_res_a_tubulin_image(name):
    '''
    Returns the a_tubulin image in the Low_Res_Input_Images folder as a 2d array
    
    '''
    parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    image_path = os.path.join(parent, "Low_Res_Input_Images", "a_tubulin", name)
    
    return io.imread(image_path)

def get_low_res_pattern_image(name):
    '''
    Returns the pattern image in the Low_Res_Input_Images folder as a 2d array
    
    '''
    parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    image_path = os.path.join(parent, "Low_Res_Input_Images", "Pattern", name)
    
    return io.imread(image_path)


class Low_Res_Image:
    
    def __init__(self, image_file):
        self.image_file = image_file
        
    def display_labeled_image(self):
        fig, ax = plt.subplots(1, figsize = (15,15))
        
                
        # adding labels
        for i in range(len(self.objects)):
            c = plt.Circle((self.objects[i][1], self.objects[i][0]), 30, color = 'red', linewidth = 1, fill = False)
            ax.add_patch(c)
        
        
        ax.imshow(self.image_file, cmap='gray', interpolation='nearest')
        ax.set_aspect('equal')
        plt.axis('off')
        plt.show()    
        
    def segment_image(self):
        
        self.image_file = filters.gaussian(self.image_file, sigma = gaussian_std)
        
        
        block_size = 15
        val = filters.threshold_local(self.image_file, block_size, offset = -0.0001)
                
        #image_objects = np.clip(self.image_file, val, float('inf'))
        
        image_objects = self.image_file > val
        
        label_image = label(image_objects)
        properties = regionprops(label_image)
        
        fig, ax = plt.subplots(1, figsize = (15,15))
        ax.imshow(self.image_file > val, cmap='gray', interpolation='nearest')
        ax.set_aspect('equal')
        plt.axis('off')
        plt.show()    
    
        
        self.objects = []
        

        for i in range(len(properties)):
            coord = [properties[i].centroid[0], properties[i].centroid[1]]
            self.objects.append(coord)
            
            
        
        
    def transform_coord(self, image_coord):        
        image_x = float(image_coord[int(self.image_id)][0])
        image_y = float(image_coord[int(self.image_id)][1])
                
        for i in range(self.objects.shape[0]):
            x_val = self.objects[i,1]
            y_val = self.objects[i,0]
            x_coord = image_x + (image_size/2 - x_val) * pixel_size
            y_coord = image_y + (y_val - image_size/2) * pixel_size
            cell_coord.append([x_coord, y_coord, x_val, y_val, self.image_id])


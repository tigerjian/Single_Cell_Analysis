import tensorflow as tf
import numpy as np
import os,glob
import sys,argparse
from skimage import io
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt



image_size = 300
num_channels = 2
images = []

#img_path = os.path.join(os.getcwd(), 'Eval_Images', 'Image_69.tif')
parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
img_path = os.path.join(parent, "High_Res_Input_Images", "DAPI_atubulin_pattern_all_1_001_R3D_D3D_PRJ_w435.tif")
image = io.imread(img_path)

image = image.astype(np.float)

# Set the threshold for fluorescence in the green and blue channels using Otsu's method.
g_thresh = threshold_otsu(image[:,:,1])
b_thresh = threshold_otsu(image[:,:,2])

g_mean = np.mean(image[:,:,1])
b_mean = np.mean(image[:,:,2])

g_std = np.std(image[:,:,1])
b_std = np.std(image[:,:,2])

image[:,:,1] = np.clip(image[:,:,1], g_thresh + g_std, float('inf'))
image[:,:,2] = np.clip(image[:,:,2], b_thresh - b_std, float('inf'))

image[:,:,1] -= g_mean # zero centering data
image[:,:,2] -= b_mean

image[:,:,1] /= g_std # normalizing data
image[:,:,2] /= b_std

images.append(image[:,:,1:]) # leave out red channel


x_batch = images

#x_batch = images[0].reshape(1, image_size,image_size,num_channels)

## Let us restore the saved model 
sess = tf.Session()
# Step-1: Recreate the network graph. At this step only graph is created.

info_path = os.path.join(os.getcwd(),'temp30.meta')

saver = tf.train.import_meta_graph(info_path)

# Step-2: Now let's load the weights saved using the restore method.
saver.restore(sess, tf.train.latest_checkpoint('./'))

# Accessing the default graph which we have restored
graph = tf.get_default_graph()

# Now, let's get hold of the op that we can be processed to get the output.
# In the original network y_pred is the tensor that is the prediction of the network
y_pred = graph.get_tensor_by_name("y_pred:0")

## Let's feed the images to the input placeholders
x= graph.get_tensor_by_name("x:0") 
y_true = graph.get_tensor_by_name("y_true:0") 
y_test_images = np.zeros((1, 3)) 


### Creating the feed_dict that is required to be fed to calculate y_pred 
feed_dict_testing = {x: x_batch, y_true: y_test_images}
result=sess.run(y_pred, feed_dict=feed_dict_testing)
# result is of this format [probabiliy_of_rose probability_of_sunflower]
print(result)
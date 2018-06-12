from keras.layers import Input, Conv2D, Dense, Flatten, Reshape
from keras.models import Model
from keras.backend import int_shape

import numpy as np
import os
import PCA_decomp
import file

# Parameters

input_img_size = 225

# ae is short for autoencoder
ae_params = [[2, 32], # 2 x 2 kernel size, 32 filters
             [2, 64],  # 3 conv layers
             [2, 128]]

num_features = 100

def init_params(image_mat):
# =============================================================================
#     Uses a convolutional autoencoder to initialize 
#     encoder parameters
# =============================================================================
    input_img = Input(shape = (input_img_size, input_img_size, 1))

    encoder_layer = input_img
        
    for i in range(len(ae_params)):
        encoder_layer = Conv2D(ae_params[i][1], ae_params[i][0],
                               activation = 'relu',
                               padding = 'same',
                               strides = 2,
                               kernel_initializer = 'glorot_uniform'
                               )(encoder_layer)
        
    pre_flatten_shape = int_shape(encoder_layer)
            
    encoder_flatten = Flatten()(encoder_layer)
            
    feature_vec = Dense(num_features,
                        kernel_initializer = 'glorot_uniform'
                        )(encoder_flatten)
    
    decoder_layer = Dense(int_shape(encoder_flatten)[1],
                          kernel_initializer = 'glorot_uniform'
                          )(feature_vec)
    
    decoder_layer = Reshape(pre_flatten_shape)(decoder_layer)
        
    for i in range(len(ae_params) - 1, -1, -1):
        decoder_layer = Conv2D(ae_params[i][1], ae_params[i][0],
                               activation = 'relu',
                               padding = 'same',
                               strides = 2,
                               kernel_initializer = 'glorot_uniform'
                               )(decoder_layer)
        
    decoded = Conv2D(1, 2,
                     activation = 'sigmoid',
                     padding = 'same',
                     strides = 2,
                     kernel_initializer = 'glorot_uniform'
                     )
    
    autoencoder = Model(input_img, decoded)
    
    autoencoder.compile(optimizer = 'adam',
                        loss = 'binary_crossentropy')
    
    autoencoder.fit(image_mat, image_mat,
                    epochs = 5,
                    batch_size = 10,
                    shuffle = True
                    )
    
def run_DEC():
# =============================================================================
#     Runs DEC
#     (1) Parameter Initialization
#     (2) Parameter Optimization
# =============================================================================
    image_mat = []

    for i in range(1,file.num_high_res + 1):
        parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        image_path = os.path.join(parent, "High_Res_Input_Images_Processed", "DAPI_%d.tif" % i)
        if (os.path.isfile(image_path)):
            image_mat.append(PCA_decomp.get_high_res_image(image_path))
    
    
    image_mat = np.asarray(image_mat)
    image_mat = np.reshape(image_mat, (54, 225, 225, 1))
    
    
    init_params(image_mat)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
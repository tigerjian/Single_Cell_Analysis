from keras.layers import Input, Conv2D, Dense, Flatten, Reshape, Conv2DTranspose, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.backend import int_shape


import numpy as np
import os
import PCA_decomp
import file
from image_display import display_image

# Parameters

input_img_size = 256

# ae is short for autoencoder
ae_params = [[2, 32], # 2 x 2 kernel size, 32 filters
             [2, 64],  # 3 conv layers
             [2, 128]]

num_features = 1000

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
                               kernel_initializer = 'glorot_uniform',
                               strides = 2
                               )(encoder_layer)
        
        encoder_layer = MaxPooling2D(2, padding='same')(encoder_layer)
                
    pre_flatten_shape = int_shape(encoder_layer)
            
    encoder_flatten = Flatten()(encoder_layer)
            
    feature_vec = Dense(num_features,
                        kernel_initializer = 'glorot_uniform'
                        )(encoder_flatten)
    
    decoder_layer = Dense(int_shape(encoder_flatten)[1],
                          kernel_initializer = 'glorot_uniform'
                          )(feature_vec)
    
    decoder_layer = Reshape(pre_flatten_shape[1:])(decoder_layer)
        
    for i in range(len(ae_params) - 2, -1, -1):
        decoder_layer = Conv2DTranspose(ae_params[i][1], ae_params[i][0],
                               activation = 'relu',
                               padding = 'same',
                               kernel_initializer = 'glorot_uniform',
                               strides = 2
                               )(decoder_layer)
        
        decoder_layer = UpSampling2D(2)(decoder_layer)
        
        

    decoded = Conv2DTranspose(1, 2,
                     activation = 'relu',
                     padding = 'same',
                     kernel_initializer = 'glorot_uniform',
                     strides = 2
                     )(decoder_layer)
    
    decoded = UpSampling2D(2)(decoded)
    
    autoencoder = Model(input_img, decoded)    
    
    autoencoder.compile(optimizer = 'adam',
                        loss = 'mean_squared_error')
    
    
    autoencoder.fit(image_mat, image_mat,
                    epochs = 1000,
                    batch_size = 100,
                    shuffle = True
                    )
    
    decoded_imgs = autoencoder.predict(image_mat)
    
    for i in range(5):
        display_image(image_mat[i].reshape(256,256))
        display_image(decoded_imgs[i].reshape(256,256))
    
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
            img = PCA_decomp.get_high_res_image(image_path)
            
            z = np.zeros((256,256))
            
            z[:225, :225] = img            
            image_mat.append(z)
    
    
    image_mat = np.asarray(image_mat)
    image_mat = np.reshape(image_mat, (54, 256, 256, 1))
    
    
    init_params(image_mat)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
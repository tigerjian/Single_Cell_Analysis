from keras.layers import Input, Conv2D, Dense, Flatten, Reshape, Conv2DTranspose, MaxPooling2D, UpSampling2D, Dropout
from keras.models import Model
from keras.backend import int_shape
from keras.engine.topology import Layer, InputSpec
import keras.backend as K
from keras.datasets import mnist


import numpy as np
import os
import PCA_decomp
import file
from image_display import display_image
import tSNE
from skimage.exposure import rescale_intensity
from sklearn.cluster import KMeans


# Parameters

input_img_size = 256

# ae is short for autoencoder
ae_params = [[2, 8],
             [2, 8],
             [2, 8]]

num_features = 100

num_clusters = 4

tSNE_perp = 20

init_params_epoch = 100
init_params_batch_size = 25

num_opt_iter = 100

class ClusteringLayer(Layer):

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight((self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def center_norm(image):
    m = np.mean(image)
    stdev = np.std(image)
    
    return (image - m)/stdev

def rev_center_norm(image):
    rescaled = rescale_intensity(image, (0, 255))
    print(np.mean(rescaled))
    return rescaled

def create_autoencoder():
# =============================================================================
#     This function creates the a Kera autoencoder model
# =============================================================================
    input_img = Input(shape = (input_img_size, input_img_size, 1))

    encoder_layer = input_img
    
    global dropout
        
    for i in range(len(ae_params)):
        encoder_layer = Conv2D(ae_params[i][1], ae_params[i][0],
                               activation = 'relu',
                               padding = 'same',
                               kernel_initializer = 'glorot_uniform',
                               strides = 2
                               )(encoder_layer)
        
        #encoder_layer = MaxPooling2D(2, padding='same')(encoder_layer)
                
    pre_flatten_shape = int_shape(encoder_layer)
            
    encoder_flatten = Flatten()(encoder_layer)
    
    encoder_flatten = Dropout(0.1)(encoder_flatten)
            
    feature_vec = Dense(num_features,
                        kernel_initializer = 'glorot_uniform'
                        )(encoder_flatten)
    
    feature_vec = Dropout(0.1)(feature_vec)
    
    decoder_layer = Dense(int_shape(encoder_flatten)[1],
                          kernel_initializer = 'glorot_uniform'
                          )(feature_vec)
    
    decoder_layer = Dropout(0.1)(decoder_layer)
    
    decoder_layer = Reshape(pre_flatten_shape[1:])(decoder_layer)
        
    for i in range(len(ae_params) - 2, -1, -1):
        decoder_layer = Conv2DTranspose(ae_params[i][1], ae_params[i][0],
                               activation = 'relu',
                               padding = 'same',
                               kernel_initializer = 'glorot_uniform',
                               strides = 2
                               )(decoder_layer)
        
        #decoder_layer = UpSampling2D(2)(decoder_layer)
        

    decoded = Conv2DTranspose(1, 2,
                     activation = 'relu',
                     padding = 'same',
                     kernel_initializer = 'glorot_uniform',
                     strides = 2
                     )(decoder_layer)
    
    #decoded = UpSampling2D(2)(decoded)
    
    autoencoder = Model(input_img, decoded)    
    encoder = Model(input_img, feature_vec)
    
    return (autoencoder, encoder)
    
def init_params(image_mat):
# =============================================================================
#     Uses a convolutional autoencoder to initialize 
#     encoder parameters
# =============================================================================
    ae = create_autoencoder()
    
    autoencoder = ae[0]
    encoder = ae[1]
    
    autoencoder.compile(optimizer = 'adam',
                        loss = 'mean_squared_error')
    
    
    autoencoder.fit(image_mat, image_mat,
                    epochs = init_params_epoch,
                    batch_size = init_params_batch_size,
                    shuffle = True
                    )
    
#    parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
#    path = os.path.join(parent, "Saved_Models", "Model_1.h5")
#    
#    encoder.save_weights(path)
    
    decoded_imgs = autoencoder.predict(image_mat)
    encoded_feats = encoder.predict(image_mat)
        
    for i in range(1):
        tSNE.apply_tSNE(encoded_feats, num_clusters, tSNE_perp, None, True)
    
#    for i in range(10):
#        display_image(rev_center_norm(image_mat[i].reshape(256,256)))
#        display_image(rev_center_norm(decoded_imgs[i].reshape(256,256)))
        
    return (autoencoder, encoder)

def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T
        
def opt_params(ae, image_mat):
    autoencoder = ae[0]
    encoder = ae[1]
    
    clustering_layer = ClusteringLayer(num_clusters, name = 'clustering')(encoder.output)
    opt_model = Model(inputs = encoder.input, outputs = [clustering_layer, autoencoder.output])
    opt_model.compile(loss = ['kld', 'mse'],
                      loss_weights = [0.1, 1],
                      optimizer = 'adam')
    
    kmeans = KMeans(n_clusters = num_clusters, n_init = 20)
    pred_clusters = kmeans.fit_predict(encoder.predict(image_mat))
    opt_model.get_layer(name = 'clustering').set_weights([kmeans.cluster_centers_])
    
    print(opt_model.metrics_names)
    
    q, _ = opt_model.predict(image_mat, verbose = 0)
    p = target_distribution(q) 

    for i in range(num_opt_iter):
        print("Iteration # %d" % (i + 1))
        if (i % 5):
            q, _ = opt_model.predict(image_mat, verbose = 0)
            p = target_distribution(q) 
            
        loss = opt_model.train_on_batch(x = image_mat,
                                         y=[p, image_mat])

        print(loss)
        
    q, _ = opt_model.predict(image_mat, verbose = 0)
    pred_clusters = q.argmax(1)
    
#    for i in range(len(pred_clusters)):
#        print("Cluster for # %d:" % (i + 1), pred_clusters[i])
    
    encoded_feats = encoder.predict(image_mat)
    
    cluster_hist = np.zeros(num_clusters)
    
    for i in range(len(pred_clusters)):
        cluster_hist[pred_clusters[i]] += 1
        
    print(cluster_hist)
            
    for i in range(5):
        tSNE.apply_tSNE(encoded_feats, num_clusters, tSNE_perp, pred_clusters, False)
    
    


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
            
            img = center_norm(img)
            
            image_mat.append(img)
            
    image_mat = np.asarray(image_mat)
    image_mat = np.reshape(image_mat, (file.num_high_res - 1, 256, 256, 1))
    
#    (x_train, _), (x_test, _) = mnist.load_data()
#    x_test = x_test.astype('float32') / 255.
#    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
#    x_test = x_test[:1000]
    
    ae = init_params(image_mat)
    opt_params(ae, image_mat)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
import math
import keras
import tensorflow as tf
from keras.backend import int_shape
from tensorflow.keras import layers, backend

class KernelModels:
    """ This class contains NN-based Keras models that modeling/simulating a series of computational kernels. """
    def __init__(self, model_name):
        self.model_name = model_name

    def get_model(self):
        """ Return corresponding model function object """
        func = getattr(self, self.model_name)
        assert callable(func), \
            f" Kernel model name: {self.model_name} not found. "
        return func

    @staticmethod
    def minimum_2d(in_shape, out_shape):
        """ This function return s minimum kernel model for latency testing purpose only. """
        inputs = keras.Input(shape=in_shape+(1,))
        x = layers.Conv2D(filters=1, kernel_size=1, padding='same', use_bias=False)(inputs)
        outputs = x
        return keras.Model(inputs, outputs)
        

    @staticmethod
    def sobel_2d(in_shape, out_shape):
        """ This function returns a NN-based Sobel model that simulates Sobel edge detection behavior. """
        encoded_dim = 16
        init = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=2022)
        inputs = keras.Input(shape=in_shape+(1,))
        x = layers.Conv2D(filters=2, kernel_size=3, padding='same', activation='relu', use_bias=False, kernel_initializer=init)(inputs)
        x = layers.Conv2D(filters=encoded_dim, kernel_size=3, padding='same', activation='relu', use_bias=False, kernel_initializer=init)(x)
        x = layers.Conv2D(filters=1, kernel_size=3, padding='same', activation='relu', use_bias=False, kernel_initializer=init)(x)
        outputs = x
        return keras.Model(inputs, outputs)
    
    @staticmethod
    def npu_sobel_2d(in_shape, out_shape):
        """ This function returns the NPU sobel model that simulates Sobel edge detection behavior. """
        #init = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=2022)
        init = keras.initializers.Constant(0.1)
        def expand_dims(x):
            """ function callable wrapper for channel-wise dense. """
            x = tf.expand_dims(x, -1)
            w = tf.Variable(tf.shape(x))
            r = tf.matmul(x, w)
            return r

        inputs = keras.Input(shape=in_shape+(1,))
        x = inputs
        for i in range(1):
            x = layers.Conv2D(filters=16, 
                          kernel_size=3, 
                          padding='same', 
                          activation='sigmoid', 
                          use_bias=False)(x)
        #x = layers.Lambda(expand_dims)(x)
        x = layers.Conv2D(filters=1, 
                          kernel_size=1, 
                          padding='same', 
                          activation='sigmoid', 
                          use_bias=False)(x)
        outputs = x
        return keras.Model(inputs, outputs)
    
    @staticmethod
    def mean_2d(in_shape, out_shape):
        """ This functions returns a NN-based mean filter model. """
        inputs = keras.Input(shape=in_shape+(1,))
        x = layers.Conv2D(filters=1, kernel_size=(3,3), padding='same', use_bias=False)(inputs)
        outputs = x
        return keras.Model(inputs, outputs)

    @staticmethod
    def laplacian_2d(in_shape, out_shape):
        """ This function returns a NN-based Laplacian filter model."""
        #encoded_dim = 16
        #inputs = keras.Input(shape=in_shape+(1,))
        #x = layers.Conv2D(filters=2, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
        #x = layers.Conv2D(filters=encoded_dim, kernel_size=(3, 3), padding='same', activation='relu')(x)
        #x = layers.Conv2D(filters=1, kernel_size=(3, 3), padding='same', activation='relu')(x)
        #outputs = x
        #return keras.Model(inputs, outputs)
        
        encoded_dim = 16
        init = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=2022)
        inputs = keras.Input(shape=in_shape+(1,))
        x = layers.Conv2D(filters=encoded_dim, kernel_size=3, padding='same', activation='relu', kernel_initializer=init)(inputs)
        x = layers.Conv2D(filters=encoded_dim, kernel_size=3, padding='same', activation='relu', kernel_initializer=init)(x)
        x = layers.Conv2D(filters=1, kernel_size=3, padding='same', activation='sigmoid', kernel_initializer=init)(x)
        outputs = x
        return keras.Model(inputs, outputs)

    @staticmethod
    def fft_2d(in_shape, out_shape):
        """ This function returns a NN-based convolveFFT2D model. """
        #encoded_dim = 4
        inputs = keras.Input(shape=in_shape+(1,))
        x = layers.Conv2D(filters=1, kernel_size=(7, 7), padding='same', use_bias=False)(inputs)
        outputs = x
        return keras.Model(inputs, outputs)
        
        #encoded_dim = 16
        #init = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=2022)
        #inputs = keras.Input(shape=in_shape+(1,))
        #x = layers.Conv2D(filters=2, kernel_size=3, padding='same', activation='relu', kernel_initializer=init)(inputs)
        #x = layers.Conv2D(filters=encoded_dim, kernel_size=3, padding='same', activation='relu', kernel_initializer=init)(x)
        #x = layers.Conv2D(filters=1, kernel_size=3, padding='same', activation='relu', kernel_initializer=init)(x)
        #outputs = x
        #return keras.Model(inputs, outputs)
    
    @staticmethod
    def dct8x8_2d(in_shape, out_shape):
        """ This function returns a NN-based dct8x8 model. """
        inputs = keras.Input(shape=in_shape+(1,))
        x = layers.Conv2D(filters=1, kernel_size=(3, 3), padding='same')(inputs)
        outputs = x
        return keras.Model(inputs, outputs)

    @staticmethod
    def hotspot_2d(in_shape, out_shape):
        """ This function returns a NN-based hotspot model. """
        encoded_dim = 2
        inputs = keras.Input(shape=(in_shape[0]*2, in_shape[1])+(1,))
        # temp slice part
        x = tf.slice(inputs, [0, 0, 0, 0], [1, in_shape[0], in_shape[1], 1])
        x = layers.Conv2D(filters=encoded_dim, kernel_size=(3,3), padding='same', activation='relu')(x)
        
        # power slice part
        y = tf.slice(inputs, [0, in_shape[0], 0, 0], [1, in_shape[0], in_shape[1], 1])
        y = layers.Conv2D(filters=encoded_dim, kernel_size=(1,1), padding='same', activation='relu')(y)
        
        #x = layers.Add()([x, y])
        x = layers.Conv2D(filters=1, kernel_size=(3,3), padding='same', activation='relu')(x)
        outputs = x
        return keras.Model(inputs, outputs)

    @staticmethod
    def srad_2d(in_shape, out_shape):
        """ This function returns a NN-based srad model. """
        encoded_dim = 16
        inputs = keras.Input(shape=in_shape+(1,))
        x = layers.Conv2D(filters=encoded_dim, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
        x = layers.Conv2D(filters=1, kernel_size=(3, 3), padding='same', activation='relu')(x)
        outputs = x
        return keras.Model(inputs, outputs)

    @staticmethod
    def blackscholes_2d(in_shape, out_shape):
        """ This function returns a NN-based blackscholes model. """
        encoded_dim = 16
        inputs = keras.Input(shape=(in_shape[0]*3, in_shape[1])+(1,))
        # slice 1
        x1 = tf.slice(inputs, [0, 0, 0 ,0], [1, in_shape[0], in_shape[1], 1])
        # slice 2
        x2 = tf.slice(inputs, [0, in_shape[0], 0 ,0], [1, in_shape[0], in_shape[1], 1])
        # slice 3
        x3 = tf.slice(inputs, [0, in_shape[0]*2, 0 ,0], [1, in_shape[0], in_shape[1], 1])
    
        init = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=2022)

        for i in range(2):
            x1 = layers.Conv2D(filters=encoded_dim, kernel_size=(1, 1), padding='same', activation='sigmoid', kernel_initializer=init)(x1)
            x2 = layers.Conv2D(filters=encoded_dim, kernel_size=(1, 1), padding='same', activation='sigmoid', kernel_initializer=init)(x2)
            x3 = layers.Conv2D(filters=encoded_dim, kernel_size=(1, 1), padding='same', activation='sigmoid', kernel_initializer=init)(x3)
        
        x = layers.Add()([x1, x2])
        x = layers.Add()([x, x3])
    
        x = layers.Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer=init)(x)
        #x = tf.slice(x, [0, 0, 0 ,0], [1, in_shape[0], in_shape[1], 1])
        
        outputs = x
        return keras.Model(inputs, outputs)
    
    @staticmethod
    def dwt_2d(in_shape, out_shape):
        """ This function returns a NN-based dwt model. """
        
        encoded_dim = 16
        inputs = keras.Input(shape=in_shape+(1,))
        x = layers.Conv2D(filters=encoded_dim, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
        x = layers.Conv2D(filters=1, kernel_size=(3, 3), padding='same', activation='relu')(x)
        outputs = x
        return keras.Model(inputs, outputs)

    @staticmethod
    def histogram_2d(in_shape, out_shape):
        """ This function returns a NN-based hist256 model. """
        inputs = keras.Input(shape=in_shape+(1,))
        iters = int(math.log(in_shape[0] / 4., 4))
        pow2 = in_shape[0] / 256.
        init = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=2022)
        encoded_dim = 16
        x = inputs
        print("pow2: ", pow2)
        for i in range(iters):
            x = layers.Conv2D(filters=encoded_dim, kernel_size=4, strides=(1, 4), padding='same', activation='relu', kernel_initializer=init)(x)
        x = layers.AveragePooling2D(pool_size=(8, 2))(x)
        x = layers.Conv2D(filters=1, kernel_size=4, strides=(1, 4), padding='same', activation='relu', kernel_initializer=init)(x)

        outputs = x
        return keras.Model(inputs, outputs)

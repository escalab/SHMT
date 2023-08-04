import os
seed=2022
os.environ['PYTHONHASHSEED'] = str(seed)
import tensorflow as tf
#tf.keras.utils.set_random_seed(seed)
#tf.config.experimental.enable_op_determinism()

assert 'IS_GPGTPU_CONTAINER' in os.environ, \
           f" Kernel model generating script is not running within GPGTPU container. "
import time
import keras
import random
import argparse
import cv2 as cv
import subprocess
import numpy as np
from PIL import Image
from scipy.stats import binom
from utils.params_base import TrainParams
from utils.utils import (
        get_imgs_count, 
        get_img_paths_list,
        load_hotspot_data,
        load_blackscholes_data,
        Quality,    
)
from keras.callbacks import (
        ModelCheckpoint, 
        EarlyStopping
)
from keras.models import Sequential
from tensorflow.keras import (
        layers, 
        callbacks
)
from tensorflow.keras.optimizers import Adam
import tensorflow_model_optimization as tfmot
from utils.params_base import (
        TrainParamsBase,
        fft_2d_kernel_array
)
from kernels.kernel_models import KernelModels
from kernels.ground_truth_functions import Applications

class MyDataGen():
    """ A customized data generator """
    def __init__(self, params, target_func):
        self.batch_size         = params.batch_size
        self.in_shape           = params.in_shape
        self.out_shape          = params.out_shape
        self.num_samples        = params.num_train
        self.num_representative = params.num_representative
        self.model_name         = params.model_name
        # The ground truth function callable
        self.func               = target_func
        # data/ILSVRC/Data/DET/train/ILSVRC2014_train_0000$
        #self.input_img_paths    = get_img_paths_list("/mnt/Data/Sobel_2048/in/train/ILSVRC2014_train_0000/", '.JPEG')[:self.num_samples] 
        #self.input_img_paths    = get_img_paths_list("/mnt/Data/DET/train/ILSVRC2014_train_0000/", '.JPEG')[:self.num_samples] 
        #self.num_imgs           = len(self.input_img_paths)

    def random_input_gen(self):
        """ This function generates random samples for training input. """
        if self.model_name == "hotspot_2d":
            x = np.zeros((self.num_samples,) + (self.in_shape[0] * 2, self.in_shape[1]) + (1,), dtype="float32")
        elif self.model_name == "blackscholes_2d":
            x = np.zeros((self.num_samples,) + (self.in_shape[0] * 3, self.in_shape[1]) + (1,), dtype="float32")
        else:
            x = np.zeros((self.num_samples,) + self.in_shape + (1,), dtype="float32")

        if self.model_name == "histogram_2d":
            y = np.zeros((self.num_samples,) + (256,1) + (1,), dtype="float32")
        else:
            y = np.zeros((self.num_samples,) + self.out_shape + (1,), dtype="float32")
        
        for j in range(self.num_samples):
            if self.model_name == 'histogram_2d':
                #mu, sigma = 128, 16 # mean and standard deviation
                #s = np.random.normal(mu, sigma, self.in_shape)
                #x_slice = s.astype("uint8")
                x_slice = np.random.randint(256, size=self.in_shape, dtype="uint8") 
                tf.keras.utils.set_random_seed(seed)
                y_slice = self.func(x_slice)
                x_max = x_slice.max()
                y_max = y_slice.max()
                #print("x_slice: ", x_slice, "x shape: ", x_slice.shape)
                #print("y_slice: ", y_slice, "y shape: ", y_slice.shape)
            elif self.model_name == 'fft_2d':
                np.random.seed(j)
                # use the same input data range 0 ~ 15 as samples/3_Imaging/convolutionFFT2D does
                x_slice = np.random.randint(16, size=self.in_shape, dtype="uint8") 
                tf.keras.utils.set_random_seed(seed)
                y_slice = self.func(x_slice, fft_2d_kernel_array)
                x_max = 15.
                y_max = 3300.
            elif self.model_name == 'hotspot_2d':
                temp_slice, power_slice = load_hotspot_data(self.in_shape)
                x_slice = np.concatenate((temp_slice, power_slice))
                y_slice = self.func(x_slice)
                x_max = 1. #x_slice.max()
                y_max = y_slice.max()
                #print("x_max: " ,x_max, ", y_max: ", y_max)
            elif self.model_name == 'srad_2d':
                image = Image.open("/home/data/lena_gray_2Kx2K.bmp")
                image = image.resize(self.in_shape)
                x_slice = np.asarray(image).astype('float32') / 255. 
                y_slice = self.func(x_slice)
                x_max = x_slice.max()
                y_max = y_slice.max()
            elif self.model_name == "blackscholes_2d":
                a_slice, b_slice, c_slice = load_blackscholes_data(self.in_shape)
                x_slice = np.concatenate((a_slice, b_slice, c_slice))
                y_slice = self.func(x_slice)
                x_max = 1. #x_slice.max()
                y_max = 1. #y_slice.max()

                #print("x max: ", x_max, ", y max: ", y_max)
                #print("x_slice: ", x_slice, ",\n x max: ", x_slice.max(), ", x min: ", x_slice.min())
                #print("y_slice: ", y_slice, ",\n y max: ", y_slice.max(), ", y min: " , y_slice.min())

            elif self.model_name == "dwt_2d":
                image = Image.open("/home/data/lena_gray_2Kx2K.bmp")
                image = image.resize(self.in_shape)
                x_slice = np.asarray(image).astype("float32")

                y_slice = self.func(x_slice)
                x_max = x_slice.max()
                y_max = y_slice.max()
            else:
                image = Image.open("/home/data/lena_gray_2Kx2K.bmp")
                image = image.resize(self.in_shape)
                x_slice = np.asarray(image)

                y_slice = self.func(x_slice)
                x_max = x_slice.max()
                y_max = y_slice.max()

            if self.model_name == 'hotspot_2d':
                temp_slice -= temp_slice.min()
                temp_slice /= temp_slice.max()
                power_slice -= power_slice.min()
                power_slice /= power_slice.max()
                x_slice = np.concatenate((temp_slice, power_slice))
                x_slice = np.expand_dims(x_slice, axis=-1)
                y_slice = np.expand_dims(y_slice, axis=-1)
            
                x[j] = (x_slice.astype('float32')) 
                y[j] = (y_slice.astype('float32') / y_max)
            else:
                x_slice = np.expand_dims(x_slice, axis=-1)
                y_slice = np.expand_dims(y_slice, axis=-1)
            
                x[j] = (x_slice.astype('float32') / x_max) 
                y[j] = (y_slice.astype('float32') / y_max)
        
        return x, y
    
    def representative_gen(self):
        """ representative dataset generator """
        for j in range(self.num_representative):
            if self.model_name == "mean_2d" or self.model_name == "sobel_2d":
                np.random.seed(j)
                x_slice = np.random.randint(255, size=(1,) + self.in_shape, dtype="uint8")
                tf.keras.utils.set_random_seed(seed)
            elif self.model_name == "laplacian_2d":
                image = Image.open("/home/data/lena_gray_2Kx2K.bmp")
                image = image.resize(self.in_shape)
                x_slice = np.expand_dims(image, axis=0).astype("uint8")
            elif self.model_name == "dct8x8_2d":
                image = Image.open("/home/data/lena_gray_2Kx2K.bmp")
                image = image.resize(self.in_shape)
                x_slice = np.expand_dims(image, axis=0).astype("uint8")
            elif self.model_name == "srad_2d":
                image = Image.open("/home/data/lena_gray_2Kx2K.bmp")
                image = image.resize(self.in_shape)
                x_slice = np.expand_dims(image, axis=0).astype("uint8")
            elif self.model_name == "hotspot_2d":
                temp_slice, power_slice = load_hotspot_data(self.in_shape)
                temp_slice -= temp_slice.min()
                temp_slice /= temp_slice.max()
                power_slice -= power_slice.min()
                power_slice /= power_slice.max()
                x_slice = np.concatenate((temp_slice, power_slice))
                x_slice = np.expand_dims(x_slice, axis=0)
            elif self.model_name == "blackscholes_2d":
        #        np.random.seed(j)
        #        x_slice = np.random.randint(255, size=(1,) + (self.in_shape[0]*3, self.in_shape[1]), dtype="uint8")
        #        tf.keras.utils.set_random_seed(seed)
                a_slice, b_slice, c_slice = load_blackscholes_data(self.in_shape)
                x_slice = np.concatenate((a_slice, b_slice, c_slice)).astype("uint8")
                x_slice = np.expand_dims(x_slice, axis=0)
            #elif self.model_name == "histogram_2d":
            #    mu, sigma = 128, 16 # mean and standard deviation
            #    s = np.random.normal(mu, sigma, (1,) + self.in_shape)
            #    x_slice = s.astype("uint8")
            else:
                np.random.seed(j)
                x_slice = np.random.randint(255, size=(1,) + self.in_shape, dtype="uint8")
                tf.keras.utils.set_random_seed(seed)

            x_slice = np.expand_dims(x_slice, axis=-1)
            x = x_slice.astype('float32')
            yield [x]

def gpu_setup():
    """ GPU setup """
    physical_devices = tf.config.list_physical_devices('GPU')
    #tf.config.experimental.set_memory_growth(physical_devices[0], True)
    assert tf.test.is_built_with_cuda()

def get_funcs(model_name):
    """ Get target function as ground truth generator and kernel model in Keras for training."""
    my_kernel_model = KernelModels(model_name)
    my_application  = Applications(model_name)
    model = my_kernel_model.get_model()
    app = my_application.get_func()
    return app, model

def model_lr(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * 0.995 

def train(params, train_from_scratch, kernel_model, random_input_gen, qat):
    """ The main training script """
    gpu_setup()

    if not os.path.exists(params.saved_model_dir+"/saved_model.pb") or train_from_scratch:
        print("===== Train from scratch... =====")
        model = kernel_model(params.in_shape, params.out_shape)
    else:
        loaded_model = tf.keras.models.load_model(params.saved_model_dir)
        target_model = kernel_model(params.in_shape, params.out_shape)
        if target_model.get_config() != loaded_model.get_config():
            print("===== WARNING: loaded pre-train weight is different from model config of the target model. Fall back to train from scratch. =====")
            model = target_model
        else:
            print("===== Start from pre-trained weights. =====")
            model = loaded_model

    # enable quantization-aware training if the flag is set.
    if qat == True:
        quantize_model = tfmot.quantization.keras.quantize_model
        #tfmot.quantization.keras.quantize_apply(annotated_model)
        model = quantize_model(model)
        print("===== QAT mode is set. =====")

    model.summary()

    X_train, Y_train = random_input_gen()

    model.compile(optimizer=params.optimizer, 
              loss=params.loss, 
              metrics=params.metrics)
    
    early = EarlyStopping(min_delta=params.min_delta, 
                          patience=params.patience, 
                          verbose=params.verbose, 
                          mode=params.mode)

    cp_callback = keras.callbacks.ModelCheckpoint(filepath=params.checkpoint_path, 
                                              save_weights_only=params.save_weights_only, 
                                              save_best_only=params.save_best_only,
                                              verbose=params.verbose)

    lr_scheduler = keras.callbacks.LearningRateScheduler(model_lr)

    print("model.fit starting...")
    hist = model.fit(X_train,
                     Y_train,
                 epochs=params.epochs, 
                 batch_size=params.batch_size,
                 validation_split=0.1,
                 shuffle=False,
                 max_queue_size=params.max_queue_size,
                 use_multiprocessing=params.use_multiprocessing,
                 workers=params.workers,
                 verbose=params.verbose,
                 callbacks=[early, cp_callback, lr_scheduler])

    tf.keras.models.save_model(model, params.saved_model_dir)
    print(f"model \"{args.model}\" saved at {params.saved_model_dir}.")

def calc_metrics(Y_ground_truth, Y_predict, logfile):
    quality = Quality(Y_ground_truth, Y_predict)
    print("error_rate      : ", quality.error_rate(), " %")
    print("error_percentage: ", quality.error_percentage(), " %")
    print("RMSE_percentage : ", quality.rmse_percentage(), " %")
    print("SSIM            : ", quality.ssim())
    print("PNSR            : ", quality.pnsr(), " dB")
    with open(logfile, 'a') as f:
        line = ","+str(quality.error_rate())+\
               ","+str(quality.error_percentage())+\
               ","+str(quality.rmse_percentage())+\
               ","+str(quality.ssim())+\
               ","+str(quality.pnsr())+",\n"
        f.write(line)

def pre_quantize_test(params, target_func, logfile):
    print("starting pre_quantize_test...")
    # get ground truth
    if params.model_name == 'fft_2d':
        X_test = np.random.randint(16, size=params.in_shape, dtype="uint8") 
        Y_ground_truth = target_func(X_test, fft_2d_kernel_array)
        x_scale = 15.
        y_scale = 3300.
    elif params.model_name == 'hotspot_2d':
        temp_slice, power_slice = load_hotspot_data(params.in_shape)
        X_test = np.concatenate((temp_slice, power_slice))
        Y_ground_truth = target_func(X_test)
        temp_slice -= temp_slice.min()
        temp_slice /= temp_slice.max()
        power_slice -= power_slice.min()
        power_slice /= power_slice.max()
        X_test = np.concatenate((temp_slice, power_slice))
        x_scale = 1.
        y_scale = 343.76224
    elif params.model_name == 'srad_2d':
        image = Image.open(params.lenna_path)
        image = image.resize(params.in_shape)
        X_test = np.asarray(image).astype('float32') / 255.
        Y_ground_truth = target_func(X_test)
        x_scale = 1.
        y_scale = 1.
    elif params.model_name == "dwt_2d":
        image = Image.open(params.lenna_path)
        image = image.resize(params.in_shape)
        X_test = np.asarray(image).astype('uint8') 
        Y_ground_truth = target_func(np.asarray(image).astype('float32'))
        x_scale = 255.
        y_scale = 255.
    elif params.model_name == "blackscholes_2d":
        a_slice, b_slice, c_slice = load_blackscholes_data(params.in_shape)
        X_test = np.concatenate((a_slice, b_slice, c_slice))
        Y_ground_truth = target_func(X_test)
        x_scale = 1.
        y_scale = 1.
    elif params.model_name == "histogram_2d":
        #mu, sigma = 128, 16 # mean and standard deviation
        #s = np.random.normal(mu, sigma, params.in_shape)
        #X_test = s.astype("float32")
        X_test = np.random.randint(255, size=params.in_shape, dtype="uint8").astype("float32") 
        tf.keras.utils.set_random_seed(seed)
        Y_ground_truth = target_func(X_test)
        x_scale = 1.
        y_scale = 1.
    else:
        image = Image.open(params.lenna_path)
        image = image.resize(params.in_shape)
        X_test = np.asarray(image).astype('uint8') 
        Y_ground_truth = target_func(np.asarray(image).astype('uint8'))
        x_scale = 255.
        y_scale = 255.
    
    print("X.shape: ", X_test.shape, ", dtype: ", X_test.dtype, ", max: ", X_test.max(), ", min: ", X_test.min())
    print(X_test)

    # get model and peek model weights
    np.set_printoptions(edgeitems=5, precision=3, linewidth=120)
    print("loading model")
    model = tf.keras.models.load_model(params.saved_model_dir)
    print("model weights:")
    print(model.get_weights())

    X_test = np.expand_dims(X_test, axis=(0, len(params.in_shape)+1))
    
    print("start to evaluate...")
    # get model prediction
    X_test = (X_test.astype('float32') / x_scale) 
    model.summary()
    #print(X_test[0][:,:,0])
    print("X_test:")
    print(X_test)
    print("X_test.shape: ", X_test.shape)
    Y_predict = model.predict(X_test, batch_size=1)
    Y_predict = np.asarray(Y_predict)
    print("Y_predict raw:", Y_predict)
    Y_predict = (Y_predict)  * y_scale
    if params.model_name == "hotspot_2d":
        Y_predict = (Y_predict)  + 0

    Y_predict = np.squeeze(Y_predict, axis=0)
    Y_predict = np.squeeze(Y_predict, axis=-1)

    print("Y_predict.shape: ", Y_predict.shape, ", dtype: ", Y_predict.dtype, ", max: ", Y_predict.max(), ", min: ", Y_predict.min())
    print(Y_predict)

    print("Y_ground_truth.shape: ", Y_ground_truth.shape, ", dtype: ", Y_ground_truth.dtype, ", max: ", Y_ground_truth.max(), ", min: ", Y_ground_truth.min())
    print(Y_ground_truth)

    with open(logfile, 'a') as f:
        f.write("----- trained fp32 model quality: -----,error_rate,error%,RMSE%,SSIM,PNSR\n")
    calc_metrics(Y_ground_truth, Y_predict, logfile) 

    if params.model_name != "histogram_2d":
        cv.imwrite("./ground_truth.png", Y_ground_truth)
        cv.imwrite("./predict.png", Y_predict)

def pre_edgetpu_compiler_tflite_test(params, target_func, logfile):
    """ This function test run the pre edgetpu_copiler compiled tflite model on CPU. """
    print("starting pre_edgetpu_compiler_tflite_test...")
    # get ground truth
    if params.model_name == 'fft_2d':
        X_test = np.random.randint(16, size=params.in_shape, dtype="uint8") 
        X_test = (X_test * (255./16.)).astype("uint8") # scale 0~15 to 0~255
        Y_ground_truth = target_func(X_test, fft_2d_kernel_array)
        x_scale = 15.
        y_scale = 3300. * (16./255.)
    elif params.model_name == 'hotspot_2d':
        temp_slice, power_slice = load_hotspot_data(params.in_shape)
        X_test = np.concatenate((temp_slice, power_slice))
        Y_ground_truth = target_func(X_test.astype("float32"))
        
        temp_slice -= temp_slice.min()
        temp_slice /= temp_slice.max()
        power_slice -= power_slice.min()
        power_slice /= power_slice.max()
        X_test = np.concatenate((temp_slice, power_slice))
        X_test = (X_test * 255.).astype('uint8')
        x_scale = 255.
        y_scale = 343.76224 / 255.
    elif params.model_name == 'srad_2d':
        x_scale = 1.
        y_scale = 1./255.
        image = Image.open(params.lenna_path)
        image = image.resize(params.in_shape)
        X_test = np.asarray(image).astype('float32') / 255.
        Y_ground_truth = target_func(X_test)
        X_test = np.asarray(image).astype('uint8')
    elif params.model_name == "blackscholes_2d":
        a_slice, b_slice, c_slice = load_blackscholes_data(params.in_shape)
        X_test = np.concatenate((a_slice, b_slice, c_slice))
        Y_ground_truth = target_func(X_test)
        X_test = X_test.astype("uint8")
        print("X_test max: ", X_test.max(), ", min: ", X_test.min())
        x_scale = 1.
        y_scale = 29./98. #29./255.
    elif params.model_name == "dwt_2d":
        x_scale = 255.
        y_scale = 1.
        image = Image.open(params.lenna_path)
        image = image.resize(params.in_shape)
        X_test = np.asarray(image).astype('uint8') 
        Y_ground_truth = target_func(np.asarray(image).astype('float32'))
    elif params.model_name == "histogram_2d":
        #mu, sigma = 128, 16 # mean and standard deviation
        #s = np.random.normal(mu, sigma, params.in_shape)
        #X_test = s.astype("uint8")
        X_test = np.random.randint(256, size=params.in_shape, dtype="uint8") 
        X_test = X_test.astype("uint8") 
        Y_ground_truth = target_func(X_test)
        x_scale = 1.
        y_scale = 1.
    else:
        x_scale = 255.
        y_scale = 1.
        image = Image.open(params.lenna_path)
        image = image.resize(params.in_shape)
        X_test = np.asarray(image).astype('uint8') 
        Y_ground_truth = target_func(np.asarray(image).astype('uint8'))

    interpreter = tf.lite.Interpreter(model_path=params.tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
   
    #if input_details['dtype'] == np.uint8:
    #    input_scale, input_zero_point = input_details["quantization"]
    #    X_test = X_test / input_scale + input_zero_point

    X_test = np.expand_dims(X_test, axis=-1)
    X_test = np.expand_dims(X_test, axis=0)
    interpreter.set_tensor(input_details["index"], X_test)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]

    Y_predict = interpreter.get_tensor(output_details["index"])[0]
    Y_predict = np.squeeze(Y_predict, axis=-1) * y_scale

    print("X_test: ", np.squeeze(np.squeeze(X_test, axis=0), axis=-1))
    print("Y_ground_truth: ", Y_ground_truth, ", max: ", Y_ground_truth.max(), ", min: ", Y_ground_truth.min(), ", shape: ", Y_ground_truth.shape)
    print("Y_predict: ", Y_predict, ", max: ", Y_predict.max(), ",min: ", Y_predict.min(), ",shape: ", Y_predict.shape)
    input_scale, input_zero_point = input_details["quantization"]
    print("input_scale: ", input_scale, ", input_zero_point: ", input_zero_point)
    with open(logfile, 'a') as f:
        f.write("----- pre-edgetpu_compiler uint8 tflite model quality: -----,error_rate,error%,RMSE%,SSIM,PNSR\n")
    calc_metrics(Y_ground_truth, Y_predict, logfile) 

def convert_to_tflite(params, representative_gen, target_func, qat, logfile):
    """ This function converts saved tf model to edgeTPU-compatible tflite model. """

    print("start converting to tflite setting...(QAT = ", qat, ")")
    converter = tf.lite.TFLiteConverter.from_saved_model(params.saved_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    if 1: #qat == False:
        converter.representative_dataset = representative_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

    print("converter starts converting...")
    tflite_model = converter.convert()
    with open(params.tflite_model_path, "wb") as f:
        f.write(tflite_model)
    print("pre-edgetpu_compiler tflite testing...")
    pre_edgetpu_compiler_tflite_test(params, target_func, logfile)
    print("edgetpu_compiler compiling...")
    os.system("edgetpu_compiler -s -m 13 "+params.tflite_model_path+" -o "+params.saved_model_dir)

def main(args):
    """ The main script """
    params = TrainParams(args.model, args.size)
    assert (args.model in dir(KernelModels) and args.model in dir(Applications)), \
        f" Given model name \"{args.model}\" is not supported. Check for available kernel and application implementations."
    for k, v in vars(params).items():
        print(k, ": ", v)
    target_func, kernel_model = get_funcs(args.model)
    my_data_gen = MyDataGen(params, target_func)
    if args.skip_train == False:
        train(params, args.train_from_scratch, kernel_model, my_data_gen.random_input_gen, args.qat)
    if args.skip_pre_test == False:
        pre_quantize_test(params, target_func, args.logfile)
    if args.skip_tflite == False:
        convert_to_tflite(params, my_data_gen.representative_gen, target_func, args.qat, args.logfile)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This Python script generates NN-based tflite model that simulates target function kernel.')
    parser.add_argument('--model', action='store', type=str, help='name of the kernel model for training')
    parser.add_argument('--skip_train', dest='skip_train', action='store_true', help='To skip kernel model training if saved model already exists.')
    parser.add_argument('--train_from_scratch', dest='train_from_scratch', action='store_true', help='Train from scratch flag when train is enabled.')
    parser.add_argument('--skip_pre_test', dest='skip_pre_test', action='store_true', help='To skip testing on trained fp32 model (before quantization).')
    parser.add_argument('--skip_tflite', dest='skip_tflite', action='store_true', help='To skip tflite converting.')
    parser.add_argument('--size', dest='size', action='store', type=int, help='problem size.')
    parser.add_argument('--QAT', dest='qat', action='store_true', help='to enable quantization-aware training. Default is disabled.')
    parser.add_argument('--logfile', action='store', type=str, help='log file path')

    parser.set_defaults(model="mean_2d")
    parser.set_defaults(size=2048)
    parser.set_defaults(skip_train=False)
    parser.set_defaults(train_from_scratch=False)
    parser.set_defaults(skip_tflite=False)
    parser.set_defaults(qat=False)
    parser.set_defaults(logfile="./log.csv")

    args = parser.parse_args()
    with open(args.logfile, 'a') as f:
        line = "model: "+ str(args.model)+\
               ",size: "+str(args.size)+\
               ",QAT: "+str(args.qat)+", =====,,\n"
        f.write(line)
    main(args)

import os
import math
import ctypes
import subprocess
import numpy as np
from ctypes import *
from skimage.metrics import(
        structural_similarity,
        peak_signal_noise_ratio
)
from numpy.ctypeslib import ndpointer

def get_gittop():
    """ This function returns the absolute path of current git repo root. """
    return subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], \
                            stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')

def get_imgs_count(path, ext):
    """ This function returns number of 'ext' type of files under 'path' dir. """
    return len([os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith(ext)])

def get_img_paths_list(path, ext, sort=True):
    """ This function returns list of 'ext' type of files under 'path' dir. """
    ret = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith(ext)]
    return sorted(ret) if sort == True else ret

def load_hotspot_data(in_shape):
    """ A helper function to load temp and power matrices 
    of application hotspot. """
    so_file = "/home/src/kernels/function_hotspot.so"
    lib = ctypes.cdll.LoadLibrary(so_file)
    hotspot_read_data = lib.read_data
    hotspot_read_data.argtypes = [c_int, \
                                  c_int, \
                                  ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), \
                                  ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
    hotspot_read_data.retype = None
    temp_slice  = np.empty(in_shape).astype("float32")
    power_slice = np.empty(in_shape).astype("float32")
    hotspot_read_data(in_shape[0], \
                      in_shape[1], \
                      temp_slice, power_slice)
    return temp_slice, power_slice

def load_blackscholes_data(in_shape):
    """ A helper function to load input data of blackscholes """
    so_file = "/home/src/kernels/function_blackscholes.so"
    lib = ctypes.cdll.LoadLibrary(so_file)
    blackscholes_read_data = lib.read_data
    blackscholes_read_data.argtypes = [c_int, \
                                       c_int, \
                                       ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), \
                                       ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), \
                                       ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
    blackscholes_read_data.retype = None
    a_slice  = np.empty(in_shape).astype("float32")
    b_slice  = np.empty(in_shape).astype("float32")
    c_slice  = np.empty(in_shape).astype("float32")
    blackscholes_read_data(in_shape[0], \
                           in_shape[1], \
                           a_slice, \
                           b_slice, \
                           c_slice)
    return a_slice, b_slice, c_slice


class Quality:
    """ This class collects all quality metircs for evaluating model output compared against ground truth. """
    def __init__(self, Y_ground_truth, Y_predict):
        self.true = Y_ground_truth.astype('float32')
        self.pred = Y_predict.astype('float32')

    def mean_of_true(self):
        return self.true.mean()

    def rmse(self):
        """ Root Mean Square Error """
        return math.sqrt(np.square(np.subtract(self.pred, self.true)).mean())

    def rmse_percentage(self):
        return (self.rmse() / self.mean_of_true()) * 100.0

    def error_rate(self):
        rate = np.fabs(np.subtract(self.pred, self.true)).mean()
        return (rate / self.mean_of_true()) * 100.0

    #def mape(self):
    #    """ Mean Absolute Percentage Error """
    #    return np.mean(np.abs((self.true - self.pred) / self.true)) * 100.0

    def error_percentage(self):
        return np.mean(np.fabs(np.subtract(self.pred, self.true)) > 10e-8) * 100.0

    def ssim(self):
        """ Structural Similarity Index """
        for i in range(len(self.true.shape)):
            if self.true.shape[i] < 7:
                print(__file__, " : small shape[", i, "] = ", self.true.shape[i], " will cause ValueError inside the skimage structural_similarity API call, so ssim measurement is skipped for now.")
                return None
        return structural_similarity(self.true, self.pred)

    def pnsr(self):
        """ Peak Signal Noise Ratio """
        return peak_signal_noise_ratio(self.true, self.pred, data_range=self.true.max() - self.true.min())

    


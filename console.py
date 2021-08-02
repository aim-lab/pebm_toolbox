import math
import sys
import os
import json
import numpy as np
import warnings
from scipy.io import loadmat

# Classes import
from src.pebm_pkg.Biomarkers import Biomarkers
from src.pebm_pkg.FiducialPoints import FiducialPoints
from src.pebm_pkg.Preprocessing import Preprocessing 

# Functions import
from src.pebm_pkg.fecgyn_tgen import *
from src.pebm_pkg.Intervals_duration import *
from src.pebm_pkg.Waves_characteristics import *
from src.pebm_pkg.Statistics import *


def biomarkers_intervals(signal, fiducial_points, **kwargs):
    bm = Biomarkers(signal, fiducials=fiducial_points,  **kwargs)
    intv, stat_int = bm.intervals()
    return stat_int

def biomarkers_waves(signal, fiducial_points, **kwargs):
    bm = Biomarkers(signal, fiducials=fiducial_points, **kwargs)
    waves, stat_wav = bm.waves()
    return stat_wav

def fiducial_points(signal, **kwargs):
    fp = FiducialPoints(signal, **kwargs)
    return fp.wavedet()

def preprocessing_notch(signal, **kwargs):
    pp = Preprocessing(signal, **kwargs)
    return pp.notch(kwargs['notch_freq'])

def preprocessing_bpflit(signal, **kwargs):
    pp = Preprocessing(signal, **kwargs)
    return pp.bpfilt()

def preprocessing_bsqi(signal, **kwargs):
    pp = Preprocessing(signal, **kwargs)
    return pp.bsqi()

def preprocessing_epltd(signal, **kwargs):
    pp = Preprocessing(signal, **kwargs)
    return pp.epltd()



# ignore all warning at runtime
warnings.filterwarnings("ignore")

# Internal Exceptions
class InvalidArgsException(Exception):
    pass

class FunctionCallException(Exception):
    pass

def read_signal_file(signal_file):
    with open(signal_file, "r") as _file:
        signal = [str(line).strip() for line in _file.readlines()]
        if "sample rate" in signal[0].lower():
            signal = signal[1:]
        return np.array(signal).astype(np.float)

def read_signal_vec(signal_vec):
    return np.array(signal_vec).astype(np.float)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise InvalidArgsException()
    else:
        # Command line arguments: 
        # <exe> <signal path> <fiducials points path> <function name> <args...>
        signal_path = sys.argv[1]
        # signal = read_signal_file(signal_path)
        signal = loadmat(signal_path)

        signal = signal['signal'].squeeze()
        fiducial_points_path = sys.argv[2]
        fiducial_points = loadmat(fiducial_points_path)
        
        keys = ["Pon", "P", "Poff", "QRSon", "Q", "qrs", "S", "QRSoff", "Ton", "T", "Toff", "Ttipo", "Ttipoon", "Ttipooff"]

        position = fiducial_points['fud_points'][0,0]
        all_keys = fiducial_points['fud_points'].dtype.names
        position_values =[]
        position_keys =[]
        # preprocess the position arrays
        #-----------------------------------
        for i, key in enumerate(all_keys):
            ret_val = position[i].squeeze()
            if (keys.__contains__(key)):
                ret_val[np.isnan(ret_val)] = -1
                ret_val = np.asarray(ret_val, dtype=np.int64)
                position_values.append(ret_val.astype(int))
                position_keys.append(key)
        #-----------------------------------
        features_dict = dict(zip(position_keys, position_values))


        func_name = sys.argv[3]
        func_args = json.loads(str(sys.argv[4])) if len(sys.argv)>3 else None

        # Function Call
        if func_name not in globals().keys():
            raise NotImplementedError("function {} is not implemented".format(func_name))
        else:
            func = globals()[func_name]
            try:
                if isinstance(func_args, list):
                    res = func(signal, features_dict, *func_args)
                elif isinstance(func_args, dict):
                    res = func(signal, features_dict, **func_args)
                elif func_args is None:
                    res = func(signal, features_dict)
                if isinstance(res, np.ndarray):
                    print(json.dumps(res.tolist()))
                else:
                    print(str(res))
            except Exception:
                raise FunctionCallException()
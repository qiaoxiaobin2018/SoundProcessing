import os
import time
import csv
import numpy as np
import scipy.io.wavfile as wav
import constants as c
import matplotlib.pyplot as plt
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank


fa_dir = "F:/vox_data_spem_npy/512/train/id10000_4_12_spem.npy"
x = np.load(fa_dir)
print(x.shape)
# x = x.reshape((512,300,1))
# print(x.shape)



# feat_name_list = os.listdir(fa_dir)
# print(model_name_list)
# Top_acc = 0
# Top_model = ""
# for feat_name in feat_name_list:
#         this_feat_path = os.path.join(fa_dir , feat_name)
#         x = np.load(this_feat_path)
#         print(x.shape[1])

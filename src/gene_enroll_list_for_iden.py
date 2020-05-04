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

# 设置文件读取路径，保存路径 ，语音时间长度
fa_data_dir_old = "F:/Vox_data/vox1_dev_wav/wav/"

load_file_list = "D:\Python_projects\SoundProcessing\src/train_list_iden.txt"
save_file_list = "D:\Python_projects\SoundProcessing\src\enroll_9s.csv"
save_path = "F:/vox_data_mfcc_npy\enroll_128_9-11s"
voice_min = 700
voice_max = 1201
save_list = []

# ============================================
def get_length(filepath):
    (rate, siglist) = wav.read(filepath)
    # ================================
    # feat = logfbank(siglist, rate)
    feat = mfcc(siglist, 16000, winlen=0.025, winstep=0.01, numcep=13, nfilt=26,nfft=512)
    # ================================
    length = feat.shape[0]
    reserve_length = length - (length % 100)
    return reserve_length


def normalize_frames(m, epsilon=1e-12):
    return np.array([(v - np.mean(v)) / max(np.std(v), epsilon) for v in m])


def get_feat_1(filepath):
    (rate, siglist) = wav.read(filepath)
    # ================================
    # feat = logfbank(siglist, rate)
    feat = mfcc(siglist, 16000, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=512)
    # ================================

    length = feat.shape[0]
    reserve_length = length - (length % 100)

    feat = feat[0:reserve_length, :]

    d_feat_1 = delta(feat, 2)
    d_feat_2 = delta(d_feat_1, 2)

    feat_normal = normalize_frames(feat.T)
    d_feat_1_normal = normalize_frames(d_feat_1.T)
    d_feat_2_normal = normalize_frames(d_feat_2.T)

    features = np.stack((feat_normal,d_feat_1_normal,d_feat_2_normal),axis=2)

    return features


'''
写入硬盘，并添加到文件列表
'''


# def save_mfcc(mfcc,lable):
#     # 生成文件名
#     if (lable >= 1000):
#         filename = "id1" + str(lable) + "_" + "enroll" + ".npy"
#     elif (lable >= 100):
#         filename = "id10" + str(lable) + "_" + "enroll" + ".npy"
#     elif (lable >= 10):
#         filename = "id100" + str(lable) + "_" + "enroll" + ".npy"
#     else:
#         filename = "id1000" + str(lable) + "_" + "enroll" + ".npy"
#     # 添加到新的文件列表
#     save_list.append([filename, lable])
#     # 保存到硬盘
#     save_name =  os.path.join(save_path, filename)
#     np.save(save_name,mfcc)


def save_mfcc(mfcc,lable):
    if lable == 49:
        # 生成文件名
        if (lable >= 1000):
            filename = "id1" + str(lable) + "_" + "enroll" + ".npy"
        elif (lable >= 100):
            filename = "id10" + str(lable) + "_" + "enroll" + ".npy"
        elif (lable >= 10):
            filename = "id100" + str(lable) + "_" + "enroll" + ".npy"
        else:
            filename = "id1000" + str(lable) + "_" + "enroll" + ".npy"
        # 添加到新的文件列表
        save_list.append([filename, lable])
        # 保存到硬盘
        save_name = os.path.join(save_path, filename)
        np.save(save_name, mfcc)
# ============================================


# 获取所有文件的路径、标签
iden_list = np.loadtxt(load_file_list, str,delimiter=",")
labels = np.array([int(i[1]) for i in iden_list])
voice_list = np.array([os.path.join(fa_data_dir_old, i[0]) for i in iden_list])


# 遍历语音，为每一个说话人保存一个注册语音
print("Start processing...")
total_length = len(voice_list)
save_count = 0
done = {}
for i in range(0,15000):
    done[i] = 0
for c, ID in enumerate(voice_list):
    if c % 100 == 0: print('Finish processing for {}/{}th wav.'.format(c, total_length))
    # 获取该语音的标签
    lable = labels[c]

    if lable != 49:
        continue

    # 如果此标签已经保存，直接处理下一个标签；否则，处理本标签
    if done[lable] == 1:
        continue
    else:
        length = get_length(ID)
        if length in range(voice_min,voice_max):
             '''保存此语音为注册语音，设置 done[lable] = 1'''
             mfccc = get_feat_1(ID)
             save_mfcc(mfccc,lable)
             save_count += 1
             done[lable] = 1

    # if save_count > 1:
    #     break


# 生成 CSV 文件
with open(save_file_list,'a',newline='') as f:
    csv_writer = csv.writer(f)
    # 添加标题
    csv_head = ['filename', 'speaker']
    csv_writer.writerow(csv_head)
    # 添加内容
    for l in save_list:
        csv_content = [l[0], l[1]]
        csv_writer.writerow(csv_content)

    # 关闭文件
    f.close()



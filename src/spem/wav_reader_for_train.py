import librosa
import numpy as np
from scipy.signal import lfilter, butter

import sigproc # see details: https://www.cnblogs.com/zhuimengzhe/p/10223510.html
import constants as c
import os


def load_wav(filename, sample_rate):
	audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
	audio = audio.flatten()# 按行方向降为 1 维
	return audio


def normalize_frames(m,epsilon=1e-12):
	return np.array([(v - np.mean(v)) / max(np.std(v),epsilon) for v in m])


# https://github.com/christianvazquez7/ivector/blob/master/MSRIT/rm_dc_n_dither.m
def remove_dc_and_dither(sin, sample_rate):
	if sample_rate == 16e3:
		alpha = 0.99
	elif sample_rate == 8e3:
		alpha = 0.999
	else:
		print("Sample rate must be 16kHz or 8kHz only")
		exit(1)
	sin = lfilter([1,-1], [1,-alpha], sin)
	dither = np.random.random_sample(len(sin)) + np.random.random_sample(len(sin)) - 1
	spow = np.std(dither)
	sout = sin + 1e-6 * spow * dither
	return sout


def get_fft_spectrum(filename,start,end):
	signal = load_wav(filename,c.SAMPLE_RATE)
	signal *= 2**15

	# get FFT spectrum
	signal = remove_dc_and_dither(signal, c.SAMPLE_RATE) # 数字滤波器,去除直流和颤动成分
	signal = sigproc.preemphasis(signal, coeff=c.PREEMPHASIS_ALPHA) # 对输入信号进行预加重
	frames = sigproc.framesig(signal, frame_len=c.FRAME_LEN*c.SAMPLE_RATE, frame_step=c.FRAME_STEP*c.SAMPLE_RATE, winfunc=np.hamming) # 将信号框成重叠帧
	# print("===================")
	# print(frames.shape)
	# print("===================")
	# exit(0)
	spem = sigproc.logpowspec(frames,c.NUM_FFT) # 计算语谱图
	# print("===================")
	# print(spem)
	# print("===================")
	# print(spem.shape)
	# print("===================")
	# exit(0)

	spem_norm = normalize_frames(spem.T) # 减去均值，除以标准差

	length = spem_norm.shape[1]
	reserve_length = length - (length % 100)

	# out = fft_norm[:,0:reserve_length]    # test
	out = spem_norm[:, start:end]   # train

	return out


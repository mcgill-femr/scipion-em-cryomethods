import numpy as np
from cryomethods.functions import compute_ctf_np
from cryomethods.functions import NumpyImgHandler


import os
# Obtener el directorio actual de trabajo donde se ejecuta el programa
current_directory = os.getcwd()
print("El directorio actual donde se ejecuta este programa es:", current_directory)

sampling_rate = 2  # A/px
#Nyquist frequency
nyquist_freq = 1 / (2 * sampling_rate)
N = 1000  # Matrix size
freqs = np.linspace(-nyquist_freq, nyquist_freq, N)
freqX, freqY = np.meshgrid(freqs, freqs)
freqs = np.column_stack((freqX.flatten(), freqY.flatten()))

dfu = float(10000)
dfv = float(10000)
angle = float(10000)
volt = float(300)
cs = float(2.6)
w = float(0.1)
phase_shift = float(0)
bfactor = float(250)

ctf = compute_ctf_np(freqs,dfu,dfv,angle,volt,cs,w,phase_shift,bfactor)
ctf_2d = ctf.reshape(N,N)
npIh = NumpyImgHandler()
print('ctf_calculated numpy:',ctf)
npIh.saveMrc(ctf_2d,'CTF.mrc')


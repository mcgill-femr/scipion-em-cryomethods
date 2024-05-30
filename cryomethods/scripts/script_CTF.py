import numpy as np
from cryomethods.functions import compute_ctf_np
from cryomethods.functions import NumpyImgHandler
import os
import matplotlib.pyplot as plt
import pandas as pd

output_folder = "/mnt/6ce540ce-1d8e-40a9-9ed8-b4a3d5e51a99/jvargas/data/micSimul/"
nImg = 10000
sampling_rate = 2  # A/px
N = 2000  # Matrix size
max_noise = 0.8

# Obtener el directorio actual de trabajo donde se ejecuta el programa
current_directory = os.getcwd()
print("El directorio actual donde se ejecuta este programa es:", current_directory)
print(" ")

filenames = []
dfus = []
dfvs = []
angles = []
bfactors=[]
noises=[]

minIndx = 10000
maxIndx = minIndx+nImg
for i in range(minIndx,maxIndx):
    #Parametros:
    dfv = float(np.random.uniform(low=5000, high=35000))

    if (np.random.uniform(low=0, high=1)>0.5):
        #No significant astigmatism
        dfu = dfv + float(np.random.uniform(low=0, high=500))
    else:
        dfu = dfv + float(np.random.uniform(low= dfv*0.5, high=dfv*0.65))
        #Significant astigmatism

    angle = float(np.random.uniform(low=0, high=180))
    bfactor = float(np.random.uniform(low=100, high=500))
    filename = output_folder + 'mic' + str(i) + '.mrc'
    noise = float(np.random.uniform(low=0.0, high=max_noise))

    print("mic:",filename)
    print("dfu:",dfu)
    print("dfv:",dfv)
    print("angle",angle)
    print("bfactor",bfactor)
    print("noise",noise)
    print(" ")
    print("---- ")

    filenames.append(filename)
    dfus.append(dfu)
    dfvs.append(dfv)
    angles.append(angle)
    bfactors.append(bfactor)
    noises.append(noise)

    nyquist_freq = 1 / (2 * sampling_rate)
    freqs = np.linspace(-nyquist_freq, nyquist_freq, N)
    freqX, freqY = np.meshgrid(freqs, freqs)
    freqs = np.column_stack((freqX.flatten(), freqY.flatten()))

    volt = float(300)
    cs = float(2.6)
    w = float(0.1)
    phase_shift = float(0)

    #CTF generation:
    ctf = compute_ctf_np(freqs,dfu,dfv,angle,volt,cs,w,phase_shift,bfactor)
    ctf_2d = ctf.reshape(N,N)

    #Noise matrix:
    noise_matrix = np.random.normal(loc=0.0, scale=1.0, size=(N, N))
    fft_matrix = np.fft.fftshift(np.fft.fft2(noise_matrix))

    #Matrix multiplication:
    fft_matrix = fft_matrix * ctf_2d

    # Obtener el espectro de magnitud
    magnitud = np.abs(fft_matrix)

    #Real space again:
    ifft_matrix = np.real(np.fft.ifft2(np.fft.ifftshift(fft_matrix)))
    ifft_matrix = ifft_matrix+ np.random.normal(loc=0.0, scale=noise, size=(N, N)
                     )
    npIh = NumpyImgHandler()
    npIh.saveMrc(ifft_matrix, filename)

datos = {"mics":filenames,
         "dfus":dfus,
         "dfvs":dfvs,
         "angles":angles,
         "bfactors":bfactors,
         "noise":noises}

df = pd.DataFrame(datos)
df.to_csv(output_folder+'datos_guardados.csv', index=False)


# Graficar el espectro de magnitud
#plt.figure(figsize=(6, 6))
#plt.imshow(ifft_matrix, cmap='gray')
#plt.colorbar()
#plt.title('Espectro de Magnitud de la Transformada de Fourier')
#plt.show()
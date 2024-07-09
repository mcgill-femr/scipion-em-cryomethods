import numpy as np
from cryomethods.functions import compute_ctf_np
from cryomethods.functions import NumpyImgHandler
import os
import matplotlib.pyplot as plt
import pandas as pd
import starfile

output_folder = "/mnt/6ce540ce-1d8e-40a9-9ed8-b4a3d5e51a99/jvargas/data/kk/"
nImg = 10
sampling_rate = 2  # A/px
N = 2000  # Matrix size
max_noise = 0.9

# Obtener el directorio actual de trabajo donde se ejecuta el programa
current_directory = os.getcwd()
print("El directorio actual donde se ejecuta este programa es:", current_directory)
print(" ")

indexes = []
filenames = []
dfus = []
dfvs = []
astigmatisms = []
angles = []
bfactors=[]
noises=[]

minIndx = 10000
maxIndx = minIndx+nImg

index = 0
for i in range(minIndx,maxIndx):
    #Parametros:
    dfv = float(np.random.uniform(low=5000, high=35000))

    if (np.random.uniform(low=0, high=1)>0.5):
        #No significant astigmatism
        dfu = dfv + float(np.random.uniform(low=0, high=500))
    else:
        dfu = dfv + float(np.random.uniform(low= dfv*0.5, high=dfv*0.65))
        #Significant astigmatism

    astigmatism = dfu - dfv
    angle = float(np.random.uniform(low=0, high=180))
    bfactor = float(np.random.uniform(low=100, high=500))
    filename = output_folder + 'mic' + str(i) + '.mrc'
    noise = float(np.random.uniform(low=0.0, high=max_noise))

    print("mic:",filename)
    print("dfu:",dfu)
    print("dfv:",dfv)
    print("astigmatism",astigmatism)
    print("angle",angle)
    print("bfactor",bfactor)
    print("noise",noise)
    print(" ")
    print("---- ")

    indexes.append(index)
    index = index+1
    filenames.append(filename)
    dfus.append(dfu)
    dfvs.append(dfv)
    astigmatisms.append(astigmatism)
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

datos = {
    "id":indexes,
    "enabled":1,
    "psdFile":"",
    "defocusU":dfus,
    "defocusV": dfvs,
    "defocusAngle": angles,
    "mic.Obj._filename": filenames,
    "bfactors":bfactors,
    "noise":noises}

df = pd.DataFrame(datos)
df.to_csv(output_folder+'ctfs.csv', index=False)
starfile.write(df, output_folder+'ctfs.star')

import sqlite3

#conn = sqlite3.connect('micrographs.sqlite')
#cursor = conn.cursor()
#cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
#print(cursor.fetchall())
#[('Properties',), ('Classes',), ('sqlite_sequence',), ('Objects',)]

with sqlite3.connect(output_folder+'micrographs.sqlite') as conn:
    df_Properties = pd.read_sql_query("SELECT * FROM Properties", conn)
    df_Classes = pd.read_sql_query("SELECT * FROM Classes", conn)
    df_Objects = pd.read_sql_query("SELECT * FROM Objects", conn)
    df_sqlite_sequence = pd.read_sql_query("SELECT * FROM sqlite_sequence", conn)

conn.close()
conn = sqlite3.connect(output_folder+'micrographs_new.sqlite')

df_Properties['value'][1] = nImg
df_Properties['value'][5] = True
df_Properties['value'][3] = output_folder+"micrographs.sqlite"

tmp = df_Classes['column_name']
tmp[12] = 'c02'
df_Classes['column_name'] = tmp

for i,idx in enumerate(indexes):
    tmp = df_Objects.loc[i].copy()
    tmp[6] = filenames[i]
    tmp[16] = os.path.basename(tmp[6])
    df_Objects.loc[i] = tmp
    print(df_Objects.loc[i])

print(df_Properties['value'][1])
df_Properties.to_sql('Properties', conn, if_exists='replace', index=False)
df_Objects.to_sql('Objects', conn, if_exists='replace', index=False)
df_Classes.to_sql('Classes', conn, if_exists='replace', index=False)
#df_sqlite_sequence.to_sql('sqlite_sequence', conn, if_exists='replace', index=False)
# Cerrar la conexión
conn.close()

print("close")


# Graficar el espectro de magnitud
#plt.figure(figsize=(6, 6))
#plt.imshow(ifft_matrix, cmap='gray')
#plt.colorbar()
#plt.title('Espectro de Magnitud de la Transformada de Fourier')
#plt.show()
# syntax=docker/dockerfile:1
# docker build -t scipion3cuda/cryomethods:1 .

FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,display

RUN apt-get update && apt-get upgrade -y

RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata

RUN apt-get update
RUN apt-get -y install sudo wget gcc g++ libopenmpi-dev mesa-utils openssh-client cmake libnss3 libfontconfig1 libxrender1 libxtst6 xterm libasound2 libglu1 libxcursor1 libdbus-1-3 libxkbcommon-x11-0

RUN apt-get install -y libfftw3-dev libopenmpi-dev libhdf5-dev libtiff5-dev libsqlite3-dev default-jdk git cmake
 
RUN apt-get -y install pkg-config libxft-dev libopencv-dev
RUN apt-get update

# CREATE SCIPION USER
RUN groupadd -r scipion && \
    useradd -r -m -d /home/scipion -s /bin/bash -g scipion scipion

RUN usermod -aG sudo scipion
RUN echo "abc\nabc" | passwd root
RUN echo "abc\nabc" | passwd scipion

# Prepare home and Scipion for scipion
RUN chown -R scipion:scipion /home/scipion 

USER scipion
WORKDIR /home/scipion

# INSTALL SCIPION
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /home/scipion/miniconda.sh
RUN bash /home/scipion/miniconda.sh -b
RUN /home/scipion/miniconda3/bin/conda init
RUN  ["/bin/bash", "-ci" , "python -m pip install scipion-installer"]
RUN ["/bin/bash", "-ci" , "python -m scipioninstaller /home/scipion/scipion3 -noAsk -j 4"]
RUN ["/bin/bash", "-ci" , "conda activate scipion3"]

# INSTALL PLUGINS
RUN ["/bin/bash", "-ci" , "/home/scipion/scipion3/scipion3 installp -p scipion-em-relion"]
RUN ["/bin/bash", "-ci" , "/home/scipion/scipion3/scipion3 installp -p scipion-em-gctf"]
RUN ["/bin/bash", "-ci" , "/home/scipion/scipion3/scipion3 installp -p scipion-em-gautomatch"]
RUN ["/bin/bash", "-ci" , "/home/scipion/scipion3/scipion3 installp -p scipion-em-cistem"]
RUN ["/bin/bash", "-ci" , "/home/scipion/scipion3/scipion3 installp -p scipion-em-motioncorr"]

# INSTALL CRYOMETHODS
# Set the working directory to the folder where Scipion is installed
WORKDIR /home/scipion/scipion3
# Download the cryomethods plugin:
RUN git clone https://github.com/mcgill-femr/scipion-em-cryomethods.git 
# Download cryomethods inside scipion /em folder
RUN git clone https://github.com/mcgill-femr/cryomethods.git ./software/em/cryomethods-0.1
# Install swig from anaconda
RUN ["/bin/bash", "-ci" , "conda activate scipion3 && conda install -c anaconda swig"]
# Install python3-dev
RUN echo "abc" | sudo -S apt-get -y install python3-dev
# Enter cryomethods folder
WORKDIR /home/scipion/scipion3/software/em/cryomethods-0.1
RUN echo "abc" | sudo -S cp /home/scipion/miniconda3/envs/scipion3/lib/libpython3.8.so.1.0 /usr/lib/
RUN echo "abc" | sudo -S mv /usr/lib/libpython3.8.so.1.0 /usr/lib/libpython3.8.so
# Compile alignLib
RUN ["/bin/bash", "-ci" , "conda activate scipion3 && ~/scipion3/scipion3 python alignLib/compile.py"]
# Install cryomethods plugin in scipion
RUN ["/bin/bash", "-ci" , "conda activate scipion3 && ~/scipion3/scipion3 installp -p ~/scipion3/scipion-em-cryomethods --devel"]
# Copy libraries to scipion libraries from alignLib folder (inside cryomethods folder) to scipion lib folder:
RUN ln -s ./alignLib/SpharmonicKit27/libsphkit.so ~/scipion3/software/lib/libsphkit.so 
RUN ln -s ./alignLib/frm/swig/_swig_frm.so ~/scipion3/software/lib/_swig_frm.so
# Scipion config update
WORKDIR /home/scipion
RUN ["/bin/bash", "-ci" , "echo | ./scipion3/scipion3 config --update"]

# INSTALL NANO
RUN echo "abc" | sudo -S apt-get -y install nano 
#nano ~/scipion3/config/scipion.conf

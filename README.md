# Cryomethods plugin

Cryomethods is a cryo-electron microscopy image processing plugin of Scipion framwork focussed mainly on processing macromolecular complexes showing extensive heterogeneity. Cryomethods will be integrated in the Scipion plugin manager but in the meantime you can install from the following steps:

## 1) Install CUDA: 

  #for example:
  
  $sudo apt install nvidia-cuda-toolkit=10.1.243-3

## 2) Install miniconda: 

  #link to install (https://docs.conda.io/en/latest/miniconda.html)
  
  $eval "$(/home/jvargas/.local/miniconda3/bin/conda shell.bash)"
  
  $conda activate

## 3) Install scipion3 (for a detailed description check https://scipion-em.github.io/docs/docs/scipion-modes/how-to-install.html):

  $sudo apt-get install gcc-8 g++-8 libopenmpi-dev make libopenmpi-dev python3-tk libfftw3-dev libhdf5-dev libtiff-dev libjpeg-dev libsqlite3-dev openjdk-8-jdk
  
  $export PATH=$PATH:/usr/local/cuda/bin
  
  $conda activate
  
  $export CXX_CUDA=g++-8
  
  $pip install --user scipion-installer
  
  $python -m scipioninstaller /path/where/you/want/scipion -j 4

## 4) Install Relion3

  #if you do not have installed cmake , install it $sudo apt-get install cmake 
  
  $scipion3 plugins 

  #Relion requires gcc & g++ version 8. It is very likely that you are running version 9 or newer. You have to do the following:
  
  #Use update-alternatives to change gcc to version 8 (https://linuxconfig.org/how-to-switch-between-multiple-gcc-and-g-compiler-versions-on-ubuntu-20-04-lts-focal-fossa)

  $sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8
  
  $sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
  
  $sudo update-alternatives --config gcc #(select version 8 for Relion compilation only, then change again to the previous version)

## 5) Install cryomethods:

  #Download cryomethods plugin in the folder where Scipion is installed, in my case:

  $cd  /home/jvargas/Software
  
  $git clone https://github.com/mcgill-femr/scipion-em-cryomethods.git

  #Download cryomethods inside scipion/em folder, in my case: /home/jvargas/Software/scipion/software/em
  
  $cd /home/jvargas/Software/scipion/software/em
  
  $git clone https://github.com/mcgill-femr/cryomethods.git 

  #rename cryomehods folder
  
  $mv cryomethods cryomethods-0.1
  
  #install swig from anaconda
  
  $conda install -c anaconda swig #(install swig)

  #install python3-dev
  
  $sudo apt-get install python3-dev
  
  #enter cryomethods folder
  
  $cd cryomethods-0.1

  #compile alignLib
  
  $scipion3 python alignLib/compile.py #should compile without errors

  #install cryomethods plugin in scipion
  
  $scipion3 installp -p scipion-em-cryomethods --devel

  #Copy libraries to scipion libraries from alignLib folder (inside cryomethods folder) to scipion lib folder, in my case:
  
  $cd alignLib/frm/swig

  $ln -s /home/jvargas/Software/scipion/software/em/cryomethods-0.1/alignLib/SpharmonicKit27/libsphkit.so /home/jvargas/Software/scipion/software/lib/libsphkit.so
  
  $ln -s /home/jvargas/Software/scipion/software/em/cryomethods-0.1/alignLib/frm/swig/_swig_frm.so /home/jvargas/Software/scipion/software/lib/_swig_frm.so

  #scipion config update
  
  $scipion3 config --update 

  #Check that in scipion.cof (/home/jvargas/Software/scipion/config/scipion.conf) the environment variables RELION_CRUOMETHODS_HOME, XMIPP_CRYOMETHODS_HOME and CRYOMETHODS_HOME   
  #are correctly set up, if not change the path (in my case Relion path was incorrect).

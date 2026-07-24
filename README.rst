# Cryomethods plugin

Cryomethods is a cryo-electron microscopy image processing plugin of Scipion framwork focussed mainly on processing macromolecular complexes showing extensive heterogeneity. Cryomethods will be integrated in the Scipion plugin manager but in the meantime you can install from the following steps:

# 1) Install CUDA: (i.e. $sudo apt install nvidia-cuda-toolkit=10.1.243-3)

# 2) Install miniconda: (https://docs.conda.io/en/latest/miniconda.html)

  $eval "$(/home/jvargas/.local/miniconda3/bin/conda shell.bash)"
  $conda activate

# 3) Install scipion3 (for a detailed description check https://scipion-em.github.io/docs/docs/scipion-modes/how-to-install.html):

  $sudo apt-get install gcc-8 g++-8 libopenmpi-dev make libopenmpi-dev python3-tk libfftw3-dev libhdf5-dev libtiff-dev libjpeg-dev libsqlite3-dev openjdk-8-jdk
  $export PATH=$PATH:/usr/local/cuda/bin
  $conda activate
  $export CXX_CUDA=g++-8
  $pip install --user scipion-installer
  $python -m scipioninstaller /path/where/you/want/scipion -j 4

# 4) Install Relion3

  #if you do not have installed cmake , install it $sudo apt-get install cmake 
  $scipion3 plugins 

  #Relion requires gcc & g++ version 8. It is very likely that you are running version 9 or newer. You have to do the following:
  #Use update-alternatives to change gcc to version 8 (https://linuxconfig.org/how-to-switch-between-multiple-gcc-and-g-compiler-versions-on-ubuntu-20-04-lts-focal-fossa)

  sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8
  sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
  sudo update-alternatives --config gcc #(select version 8 for Relion compilation only, then change again to the previous version)

# 5) Install cryomethods:

Install the native build tools and FFTW headers once. ``alignLib`` is written
in C; no Fortran compiler is required.

  $sudo apt-get install build-essential libfftw3-dev

SWIG is installed automatically in the Scipion Python environment by the
plugin installer.

Install the plugin through Scipion. The installer compiles ``alignLib``,
checks every compiler command, and configures its runtime library path. No
second repository, folder rename, manual compilation, or symbolic links are
needed.

  $scipion3 installp -p scipion-em-cryomethods --devel
  $scipion3 config --update

Optional validation:

  $scipion3 python -c "from cryomethods import Plugin; Plugin.setEnviron(); import frm; print('alignLib OK')"

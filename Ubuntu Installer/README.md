# Instructions for Manual install of Cryomethods plugin

These instruction are written for Ubuntu/Linux but can be easily adapted to any other Linux distribution. 
If you are using Ubuntu 22.04, we recommend to use the provided scripts for guided installation. 

## 1) Install CUDA

  For example:
  ```bash
  $sudo apt install nvidia-cuda-toolkit=10.1.243-3
  ```

## 2) Install miniconda

  link to download miniconda (https://docs.conda.io/en/latest/miniconda.html)
  
  ```bash
  $eval "$(/home/jvargas/.local/miniconda3/bin/conda shell.bash)"
  
  $conda activate
  ```

## 3) Install scipion3 

  For a detailed description check https://scipion-em.github.io/docs/docs/scipion-modes/how-to-install.html

  ```bash
  $sudo apt-get install gcc-8 g++-8 libopenmpi-dev make libopenmpi-dev python3-tk libfftw3-dev libhdf5-dev libtiff-dev libjpeg-dev libsqlite3-dev openjdk-8-jdk
  
  $export PATH=$PATH:/usr/local/cuda/bin
  
  $conda activate
  
  $export CXX_CUDA=g++-8
  
  $pip install --user scipion-installer
  
  $python -m scipioninstaller /path/where/you/want/scipion -j 4
  ```

## 4) Install Relion3

  If you do not have installed cmake, install it:
  ```bash
  $sudo apt-get install cmake 
  ```
  Then:
  ```bash
  $scipion3 plugins 
  ```
  
  Relion requires gcc & g++ version 8. It is very likely that you are running version 9 or newer. You have to use update-alternatives to change gcc to version 8 (https://linuxconfig.org/how-to-switch-between-multiple-gcc-and-g-compiler-versions-on-ubuntu-20-04-lts-focal-fossa)

  ```bash
  $sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8
  
  $sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
  
  $sudo update-alternatives --config gcc #(select version 8 for Relion compilation only, then change again to the previous version)
  ```

## 5) Install cryomethods

  Download cryomethods plugin in the folder where Scipion is installed:

  ```bash
  $cd  <path/to/installation/folder>
  
  $git clone https://github.com/mcgill-femr/scipion-em-cryomethods.git
  ```
  
  Download cryomethods inside scipion/em folder:
  
  ```bash
  $cd <path/to/installation/folder>/scipion/software/em
  
  $git clone https://github.com/mcgill-femr/cryomethods.git 
  ```
  
  Rename cryomehods folder:
  ```bash
  $mv cryomethods cryomethods-0.1
  ```
  
  Install swig from anaconda:
  ```bash
  $conda install -c anaconda swig #(install swig)
  ```
  
  Install python3-dev
  ```bash
  $sudo apt-get install python3-dev
  ```
  
  Enter cryomethods folder:
  ```bash
  $cd cryomethods-0.1
  ```
  
  Compile alignLib:
  ```bash
  $scipion3 python alignLib/compile.py #should compile without errors
  ```

  Install cryomethods plugin in scipion:
  ```bash
  $scipion3 installp -p scipion-em-cryomethods --devel
  ```
  
  Copy libraries to scipion libraries from alignLib folder (inside cryomethods folder) to scipion lib folder:
  ```bash
  $cd alignLib/frm/swig

  $ln -s <path/to/installation/folder>/scipion/software/em/cryomethods-0.1/alignLib/SpharmonicKit27/libsphkit.so <path/to/installation/folder>/scipion/software/lib/libsphkit.so
  
  $ln -s <path/to/installation/folder>/scipion/software/em/cryomethods-0.1/alignLib/frm/swig/_swig_frm.so <path/to/installation/folder>/scipion/software/lib/_swig_frm.so

  Update scipion config file:
  ```bash
  $scipion3 config --update 
  ```
  
  Check that in scipion.cof (<path/to/installation/folder>/scipion/config/scipion.conf) the environment variables RELION_CRUOMETHODS_HOME, XMIPP_CRYOMETHODS_HOME and CRYOMETHODS_HOME are correctly set up, if not change the path.

  ## 6) Use case: 3D autoclassification:
  We have included a use case to show how to run the 3D autoclassification method of CryoMethods plugin with real data. To run this example run in the command line the following line:
```bash
$scipion3 tests cryomethods.tests.test_protocols_cryomethods.Test3DAutoClasifier
```

This line downloads the test data and run the 3D autoclassification method automatically. 
To open the new created project run the following line:
```bash
$scipion3 last
```

## 7) Use case: 2D autoclassification
To run this example run in the command line the following line:
```bash
$scipion3 tests cryomethods.tests.test_protocols_cryomethods.Test2DAutoClasifier
```

This line, downloads the test data and run the 3D autoclassification method automatically. To open the new created project run the following line:
```bash
$scipion3 last
```

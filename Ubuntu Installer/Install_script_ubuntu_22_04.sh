#!/bin/bash

# 1. check that CUDA toolkit is installed
# 2. check that Conda and pip are installed
# 3. install Scipion3 (if not previously installed)
# 4. install Relion3 (if not previously installed)
# 5. install Cryomethods
# 6. test installation (optional)

########################
### Helper functions ###
########################

# aliases
PRINT='/usr/bin/printf'
BLUE='\033[1;34m'
RED='\033[0;31m' 
GREEN='\033[0;32m' 
NC='\033[0m' # No Color


# 1. check CUDA toolkit
function checkCUDA() {
    $PRINT "Checking CUDA installation... "
    nvcc -V > /dev/null 2>&1
    if [[ $? -eq 0 ]]; then
        $PRINT "${GREEN}\u2714${NC} installed\n"
    else
        $PRINT "${RED}\u2718${NC} missing\n"
        echo "Please, install cuda toolkit (e.g., sudo apt install nvidia-cuda-toolkit)"
        exit 1
    fi 
}

# 2. check Conda
function checkConda() {
    $PRINT  "Checking Conda installation... "
    conda -V > /dev/null 2>&1
    if [[ $? -eq 0 ]]; then
        $PRINT "${GREEN}\u2714${NC} installed\n"
    else
        $PRINT "${RED}\u2718${NC} missing\n"
        echo "Please, install Conda (see https://docs.conda.io/en/latest/miniconda.html)"
        exit 1
    fi 
    # check pip
    $PRINT  "Checking pip installation... "
    conda list pip | grep "pip" > /dev/null 2>&1
    if [[ $? -eq 0 ]]; then
        $PRINT "${GREEN}\u2714${NC} installed\n"
    else
        $PRINT "${RED}\u2718${NC} missing\n"
        echo "Please, install pip in your conda env"
        exit 1
    fi 
}

# 3. check Scipion3
function checkScipion(){
    $PRINT  "Checking Scipion installation... "
    if [[ -n $(conda list | grep 'scipion')  ]] 
    then
        $PRINT "${GREEN}\u2714${NC} installed\n"
        read -p 'Insert Scipion installation path (default: ~/Software/scipion): ' SCIPATH
        if [[ -z SCIPATH ]]; then
            export SCIPATH=~/Software/scipion
        else
            export SCIPATH
        fi
    else
        $PRINT "${RED}\u2718${NC} missing\n"
        echo "Scipion is missing: either install it or exit to change conda environment"
        read -p 'Do you want to install Scipion ([y]/n)? ' uservar
        if [[ $uservar != 'n' ]]
        then
            Scipion_installer
        else
            echo 'Installation exited'
            exit 1
        fi
    fi 
}

function Scipion_installer() {
    read -p 'Insert Scipion installation path (default: Software/scipion): ' SCIPATH
    if [[ -z SCIPATH ]]; then
        export SCIPATH=Software/scipion
    else
        export SCIPATH
    fi
    ## start installing packages
    sudo apt install build-essential
    # gcc 8
    sudo apt update
    wget http://mirrors.kernel.org/ubuntu/pool/universe/g/gcc-8/gcc-8_8.4.0-3ubuntu2_amd64.deb
    wget http://mirrors.edge.kernel.org/ubuntu/pool/universe/g/gcc-8/gcc-8-base_8.4.0-3ubuntu2_amd64.deb
    wget http://mirrors.kernel.org/ubuntu/pool/universe/g/gcc-8/libgcc-8-dev_8.4.0-3ubuntu2_amd64.deb
    wget http://mirrors.kernel.org/ubuntu/pool/universe/g/gcc-8/cpp-8_8.4.0-3ubuntu2_amd64.deb
    wget http://mirrors.kernel.org/ubuntu/pool/universe/g/gcc-8/libmpx2_8.4.0-3ubuntu2_amd64.deb
    wget http://mirrors.kernel.org/ubuntu/pool/main/i/isl/libisl22_0.22.1-1_amd64.deb
    sudo apt install ./libisl22_0.22.1-1_amd64.deb ./libmpx2_8.4.0-3ubuntu2_amd64.deb ./cpp-8_8.4.0-3ubuntu2_amd64.deb ./libgcc-8-dev_8.4.0-3ubuntu2_amd64.deb ./gcc-8-base_8.4.0-3ubuntu2_amd64.deb ./gcc-8_8.4.0-3ubuntu2_amd64.deb
    # g++ 8
    wget http://mirrors.kernel.org/ubuntu/pool/universe/g/gcc-8/libstdc++-8-dev_8.4.0-3ubuntu2_amd64.deb
    wget http://mirrors.kernel.org/ubuntu/pool/universe/g/gcc-8/g++-8_8.4.0-3ubuntu2_amd64.deb
    sudo apt install ./libstdc++-8-dev_8.4.0-3ubuntu2_amd64.deb ./g++-8_8.4.0-3ubuntu2_amd64.deb
    # other
    sudo apt-get install libopenmpi-dev make libopenmpi-dev python3-tk libfftw3-dev libhdf5-dev libtiff-dev libjpeg-dev libsqlite3-dev openjdk-8-jdk
    sudo apt-get install python3-opencv
    ## export variables
    export PATH=$PATH:/usr/local/cuda/bin
    export CXX_CUDA=g++-8
    ## download scipion installer
    cd ~
    pip install --user scipion-installer
    ## check that gcc is installed
    if [[ -z $(gcc --version) ]]; then
        sudo update-alternatives  --install /usr/bin/g++ g++ /usr/bin/g++-8 8
        sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8 
    fi
    ## Install scipion
    python -m scipioninstaller $SCIPATH  -j 8 

}   

# 4. check Relion3
function checkRelion(){
    $PRINT  "Checking Relion installation... "
    if [[ -n $(conda list | grep 'relion')  ]] 
    then
        $PRINT "${GREEN}\u2714${NC} installed\n"
    else
        $PRINT "${RED}\u2718${NC} missing\n"
        echo "Relion is missing: either change conda environment or install it"
        read -p 'Do you want to install Relion ([y]/n)? ' uservar
        if [[ $uservar != 'n' ]]
        then
            Relion_installer
        else
            echo 'Installation exited'
            exit 1
        fi
    fi 
}

function Relion_installer() {
    # install cmake 
    sudo apt-get install cmake 
    sudo apt-get install pkg-config
    sudo apt-get -y install libxft-dev
    sudo apt-get update

    # Relion requires gcc & g++ version 8. It is very likely that you are running version 9 or newer. You have to do the following: 
    # Use update-alternatives to change gcc to version 8 (https://linuxconfig.org/how-to-switch-between-multiple-gcc-and-g-compiler-versions-on-ubuntu-20-04-lts-focal-fossa)
    sudo update-alternatives  --install /usr/bin/g++ g++ /usr/bin/g++-8 8
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8 
    echo 'Relion requires gcc & g++ version 8. Use update-alternatives to change gcc to version 8:'
    sudo update-alternatives --config gcc 

    # install relion
    $SCIPATH/scipion3 installp -p scipion-em-relion
}   

# 5. install cryomethods
function Cryomethods_installer() {
    #  Download cryomethods plugin in the folder where Scipion is installed:
    cd $SCIPATH
    # git clone https://github.com/mcgill-femr/scipion-em-cryomethods.git ### commented only for my install, to be uncommented
    # Download cryomethods inside scipion /em folder
    cd $SCIPATH/software/em
    git clone https://github.com/mcgill-femr/cryomethods.git
    #  Rename cryomehods folder
    mv cryomethods cryomethods-0.1
    # Install swig from anaconda
    conda install -c anaconda swig
    # Install python3-dev
    sudo apt-get install python3-dev
    # Enter cryomethods folder
    cd cryomethods-0.1
    # Compile alignLib
    $SCIPATH/scipion3 python alignLib/compile.py
    # Install cryomethods plugin in scipion
    $SCIPATH/scipion3 installp -p $SCIPATH/scipion-em-cryomethods --devel
    # Copy libraries to scipion libraries from alignLib folder (inside cryomethods folder) to scipion lib folder:
    cd alignLib/frm/swig
    ln -s $SCIPATH/software/em/cryomethods-0.1/alignLib/SpharmonicKit27/libsphkit.so $SCIPATH/software/lib/libsphkit.so 
    ln -s $SCIPATH/software/em/cryomethods-0.1/alignLib/frm/swig/_swig_frm.so $SCIPATH/software/lib/_swig_frm.so
    # Scipion config update
    $SCIPATH/scipion3 config --update
}

function checkCryomethodsConfig() {
    echo -e "Check that in ${BLUE}scipion.conf${NC} the following environment variables are correctly set up:\n"
    echo -e "   ${GREEN}RELION_CRYOMETHODS_HOME${NC}"
    echo -e "   ${GREEN}XMIPP_CRYOMETHODS_HOME${NC}"
    echo -e "   ${GREEN}CRYOMETHODS_HOME${NC}"
    echo -e '\n If not, change the path. \n'
    read -p 'Do you want to check now ([y]/n)? ' uservar
    if [[ $uservar != 'n' ]];  then
        nano $SCIPATH/config/scipion.conf
    fi
}

# 6. run test 
runCryomethodsTest() {
    read -p 'Do you want to test Cryomethods installation now ([y]/n)? ' uservar
    if [[ $uservar != 'n' ]];  then
        echo 'Running 2D autoclassification... '
        $SCIPATH/scipion3 tests cryomethods.tests.test_protocols_cryomethods.Test2DAutoClasifier
    fi
}


####################
### Main program ###
####################

# 1. check CUDA toolkit is installed
checkCUDA

# # 2. check Conda is installed
checkConda

# # 3. check Scipion3 is installed
checkScipion
# activate the conda env created by scipioninstaller
conda activate scipion3

# 4. check Relion3 is installed
checkRelion

# 5. install cryomethods
Cryomethods_installer
# Check that in scipion.conf (SCIPIONPATH/config/scipion.conf) the environment variables RELION_CRYOMETHODS_HOME, XMIPP_CRYOMETHODS_HOME and CRYOMETHODS_HOME are correctly set up, if not change the path.
checkCryomethodsConfig

# 6. test installation
runCryomethodsTest

# Display final message
echo "Install Complete! "

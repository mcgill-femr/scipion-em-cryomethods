# Cryomethods plugin for Scipion

Cryomethods is a cryo-electron microscopy image processing plugin of Scipion framwork focussed mainly on processing macromolecular complexes showing extensive heterogeneity. Cryomethods will be integrated in the Scipion plugin manager but in the meantime you can install from the following steps:

## Installation

### Docker (recommended)

This is the easiest and most reliable way to install and run Cryomethods if you don't have scipion previously installed on your computer.

1. Install Docker (you don't have it already). 
To install Docker on your machine check the instructions on the official webpage https://docs.docker.com/get-docker/

2. Download the Dockerfile provided [here](./Docker/Dockerfile)

3. Build the Docker image
```bash
docker build -t scipion3cuda/cryomethods:1 .
```
4. Create on your computer a folder where you are going to store your scipion projects.

5. Download the bash script provided [here](./Docker/scipion.sh). Change the script writing in the HOSTDATAFOLDER variable the path to a folder where you are going to store your scipion projects and in the HOSTNAME variable your username.

6. Everytime you want to start the Docker container simply launch the script:
```bash
bash <path/to>/scipion.sh 
```

### CLI install

The instructions to install Cryomethods via CLI are provided [here](Ubuntu Installer/README.md).
The instructions are meant for Ubuntu but can be adapted to any other Linux distribution.

Alternatively, installation scripts are provided for Ubuntu [20.04](Ubuntu Installer/Install_script_ubuntu_20.04.sh) and [22.04](Ubuntu Installer/Install_script_ubuntu_22.04.sh). Simply launch the script, e.g.:
```bash
bash <path/to>/Install_script_ubuntu_22.04.sh
```
and a prompt will guide you through the installation process.


## Usage

To run one of the provided examples, launch inside the container:

```bash
~/scipion/scipion3 tests cryomethods.tests.test_protocols_cryomethods.Test2DAutoClasifier
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[GPL-3.0](./LICENSE)

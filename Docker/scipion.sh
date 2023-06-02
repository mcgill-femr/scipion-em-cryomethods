# Variables
HOSTUSER='<your username>'
HOSTDATAFOLDER='<path/to>/ScipionUserData'

# Do not change the following variables unless necessary
DOCKERUSER='999' # default id of the docker user outside the container (inside the container the user is named scipion)
DOCKERDATAFOLDER='/home/scipion/ScipionUserData'

# change permissions of the data folder so that it can be used inside docker (meanwhile in the host pc it must be accessed by root)
sudo chown -Rv $DOCKERUSER $HOSTDATAFOLDER > /dev/null 2>&1

# This command allows any local application to access the display server. This behavior is switched off when the application is shut down
xhost +local:

# This command runs a containerized version of scipion3. 
# If the image is not present locally, it should be automatically downloaded (this not working at the moment)
docker container run -it \
		     --rm \
		     --net host \
		     -v /tmp/.X11-unix:/tmp/.X11-unix \
		     -v $HOSTDATAFOLDER:$DOCKERDATAFOLDER \
		     -e DISPLAY \
		     -e OMPI_ALLOW_RUN_AS_ROOT=1 \
		     -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
		     --runtime=nvidia \
		     --gpus all \
		     scipion3cuda/cryomethods:1 \
		     bash	     
#		     /home/scipion/scipion3/scipion3
	
# Turn off access to the display server 	     
xhost -local:

# change back permissions of the data folder so that it can be used outside docker
sudo chown -Rv $HOSTUSER $HOSTDATAFOLDER > /dev/null 2>&1

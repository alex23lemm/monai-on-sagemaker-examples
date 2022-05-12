*** Custom Container to run MONAI training on Sagemaker ***

*** tested from Sagemaker Studio ***
Run :
- sm-docker build . --repository smstudio-monai_docker_custom:monaikernel
- create-and-update-image.sh

*** Beaware of Sagemaker Memory allocation for Docker Containers as it uses 95% of the Total Available Memory ***
# Added block (use 95% of total memory)
from psutil import virtual_memory
mem = virtual_memory()
shm_size = str(int(int(str(mem.total)[:2])*.95))+'gb'






******** NEW ******
sh build_and_push.sh ${IMAGE_NAME} ${IMAGE_TAG} $(account_id) $(region)
sh attach_image.sh ${IMAGE_NAME} ${IMAGE_TAG}
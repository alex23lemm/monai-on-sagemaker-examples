# MONAI on SageMaker Getting Started Tutorials 

This repository contains several MONAI Core on SageMaker tutorials to help researchers and data scientists to quickly get started on using MONAI Core in combination with Amazon SageMaker. 

## Disclaimer

* The content provided in this repository is for demonstration purposes and not meant for production. You should use your own discretion when using the content.
* The ideas and opinions outlined in these examples are our own and do not represent the opinions of AWS.

## Steps needed to execute the workshop code

* Have AWS CLI installed and configured. See instructions here : https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html

* Have AWS CDK installed. See instructions here : https://docs.aws.amazon.com/cdk/v2/guide/getting_started.html
I suggest to make use of python3 and virtual enviroments

* Activate the virtual enviromnent

* Go into the MonaiOnSageMakerInfrastructre folder
* By Default AWS CDK uses the settings specified during the AWS CLI installation.
  If you want to change the default account and region used by CDK application open the app.py and change there the env variable to reflect your desired values.
* From shell run the following command to deply the needed AWS resources into the chosen account and region : cdk deploy
* From shell run build_and_push.sh ${IMAGE_NAME} ${IMAGE_TAG} $(account_id) $(region)
Make sure to supply to the shell script the right account and region 
* Execute the code inside the MonaiBrainTumorSegmentation.ipynb notebook to run training
* Execute the code inside the MonaiAsyncInference/MonaiAsyncInference.ipynb notebook to run async Inference
* From shell run the following command to destroy the needed AWS resources into the chosen account and region : cdk destroy.
NOTE some resources might not be automatically destroyed. See the the CloudFormation service for details

 





# MONAI on SageMaker Getting Started Tutorials 

This repository contains several MONAI Core on SageMaker tutorials to help researchers and data scientists to quickly get started on using MONAI Core in combination with Amazon SageMaker. 

## Disclaimer

* The content provided in this repository is for demonstration purposes and not meant for production. You should use your own discretion when using the content.
* The ideas and opinions outlined in these examples are our own and do not represent the opinions of AWS.

## Steps needed to execute the workshop code

* Verify CLI with “aws —version”

* Install Nodejs:
curl -sL https://rpm.nodesource.com/setup_16.x | sudo bash -
sudo yum install nodejs
node --version

* Create AWS IAM Admin User

* Update and Configure AWS cli
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
aws configure and use text as output format

* Install the CDK
sudo npm install -g aws-cdk

* Install git
sudo yum install git -y

* Clone repository
git clone https://github.com/alex23lemm/monai-on-sagemaker-examples.git MonaiOnSagemaker

* Switch branch
cd MonaiOnSagemaker
git checkout MonaiOnSagemakerOlea

* Create python virtual environment
cd MonaiOnSagemakerInfrastructure
python3 -m venv venv

* Activate virtual environment
source venv/bin/activate

* Install python libraries
pip install -r requirements.txt

* Bootstrap cdk (https://docs.aws.amazon.com/cdk/v2/guide/bootstrapping.html)
aws bootstrap

* deploy infrastrucure
aws deploy

* Install Jupiter
pip install jupyter

* Install Jupiter, Python Environment Manager  and Python Extension Pack extension in vscode 

* deactivate python virtual environment
deactivate

* Move up one folder
cd ..

* Create a new python virtual environment
python3 -m venv venv

* Activate virtual environment
source venv/bin/activate

* select from vscode top right corner the newly created venv

* Open the notebook

* Install sagemaker sdk and boto3
pip install sagemaker
pip install boto3

* Install docker
sudo yum update
sudo yum install docker
sudo usermod -a -G docker ec2-user
id ec2-user
sudo setfacl --modify user:ec2-user:rw /var/run/docker.sock
sudo systemctl enable docker.service
sudo systemctl start docker.service
sudo systemctl status docker.service
docker version

* Run the build and push shell script giving the appropriate parameters for your user
sh build_and_push.sh monaionsagemakercontainer latest 939432307101 us-east-1

* Run the cell and VSCode will ask you to install some missing libs and please select yes

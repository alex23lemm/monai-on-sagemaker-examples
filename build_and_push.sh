#!/usr/bin/env bash

image_name=$1
tag=$2
account_id=$3
region=$4

fullname="${account_id}.dkr.ecr.${region}.amazonaws.com/${image_name}:${tag}"

echo "Image Name: ${image_name}"
echo "Tag: ${tag}"
echo "Account Id: ${account_id}"
echo "Region: ${region}"

echo "Script executed from: ${PWD}"

# Get the login command from ECR and execute it directly
aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin ${account_id}.dkr.ecr.${region}.amazonaws.com


# Base AWS DeepLearning Image
aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.${region}.amazonaws.com


# If the repository doesn't exist in ECR, create it.
aws ecr describe-repositories --repository-names ${image_name} || aws ecr create-repository --repository-name ${image_name}

# Build the docker image locally and then push it to ECR with the full name.
echo "BUILDING IMAGE WITH NAME ${image_name} AND TAG ${tag}"
#cd docker
docker build --no-cache -t ${image_name} -f Dockerfile .
docker tag ${image_name} ${fullname}

echo "PUSHING IMAGE TO ECR ECR ${fullname}"
docker push ${fullname}


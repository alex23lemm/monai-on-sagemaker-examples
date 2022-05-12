#!/usr/bin/env bash

image_name=$1
tag=$2
account_id=$3
region=$4
sagemaker_domain_id=$5
sagamaker_image_description=$6
sagemaker_image_display_name=$7
sagemaker_role_arn=$8



image_uri="${account_id}.dkr.ecr.${region}.amazonaws.com/${image_name}:${tag}"
echo
echo "Image Name: ${image_name}"
echo "Tag: ${tag}"
echo "Account Id: ${account_id}"
echo "Region: ${region}"
echo "Sagemaker Domain Id: ${sagemaker_domain_id}"
echo "Sagemaker Image Description: ${sagamaker_image_description}"
echo "Sagemaker Image Display name: ${sagemaker_image_display_name}"
echo "Sagemaker role arn: ${sagemaker_role_arn}"


echo
echo "Script executed from: ${PWD}"

echo "Image URI: ${image_uri}"

echoß

# Create Image for Studio
aws sagemaker create-image \
  --description ${sagamaker_image_description} \
  --display-name ${sagemaker_image_display_name} \
  --image-name ${image_name} \
  --role-arn ${sagemaker_role_arn}

# Create Image version for Studio
aws sagemaker create-image-version \
  --image-name ${image_name} \
  --base-image ${image_uri}

# Create AppImageConfig for this image
aws sagemaker delete-app-image-config \
  --app-image-config-name custom-monai

aws sagemaker create-app-image-config \
  --cli-input-json file://app-image-config-input.json

# Update the Domain, providing the Image and AppImageConfig
aws sagemaker update-domain \
  --domain-id ${sagemaker_domain_id} \
  --cli-input-json file://update-domain-input.json

    
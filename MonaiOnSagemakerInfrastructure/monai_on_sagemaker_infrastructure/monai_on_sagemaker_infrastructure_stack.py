from aws_cdk import (
    # Duration,
    Stack,
    aws_efs as efs,
    aws_ec2 as ec2,
    aws_ssm as ssm, 
    aws_s3 as s3, 
    aws_iam as iam
)

from constructs import Construct

class MonaiOnSagemakerInfrastructureStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # The code that defines your stack goes here
        file_system = efs.FileSystem(
            self, 
            "MonaiSagemakerEfsFileSystem",
            vpc=ec2.Vpc(self, "MonaiSagemakerVPC"),
            lifecycle_policy=efs.LifecyclePolicy.AFTER_14_DAYS,  # files are not transitioned to infrequent access (IA) storage by default
            performance_mode=efs.PerformanceMode.GENERAL_PURPOSE,  # default
            out_of_infrequent_access_policy=efs.OutOfInfrequentAccessPolicy.AFTER_1_ACCESS
        )
        file_system.connections.allow_default_port_from_any_ipv4("Allow In connection on Default port from any IPv4 address")
        
        
        sagemaker_monai_efs_id_parameter = ssm.StringParameter(
            self, 
            "SageMakerMonaiEFSIDParameter",
            allowed_pattern=".*",
            description="EFS ID for Sagemaker Monai",
            parameter_name="SageMakerMonaiEFSIDParameter",
            string_value=file_system.file_system_id,
            tier=ssm.ParameterTier.STANDARD
        )
        
        # Create Bucket for aggregate raw text files
        monaiOnSagemakerBucket = s3.Bucket(
            self,
            "MonaiOnSagemakerBucket", 
            versioned = False
        )

        monaiOnSagemakerBucketParameter = ssm.StringParameter(
            self, 
            "MonaiOnSagemakertBucketParameter",
            allowed_pattern=".*",
            description="Bucket to store datasets for the MONAI on Sagemaker DEMO",
            parameter_name="MonaiOnSagemakerBucketParameter",
            string_value=monaiOnSagemakerBucket.bucket_name,
            tier=ssm.ParameterTier.STANDARD
        )
        
        monaiOnSagemakerRole = iam.Role(
            self, 
            "MonaiOnSagemakerRole", 
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com")
        )
        monaiOnSagemakerRole.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3FullAccess")
        )
        monaiOnSagemakerRole.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEC2ContainerRegistryFullAccess")
        )
        monaiOnSagemakerRole.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AWSCodeBuildAdminAccess")
        )
        monaiOnSagemakerRole.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSNSFullAccess")
        )
        monaiOnSagemakerRole.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess")
        )
        monaiOnSagemakerRole.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerServiceCatalogProductsCodeBuildServiceRolePolicy")
        )
        monaiOnSagemakerRoleParameter = ssm.StringParameter(
            self, 
            "MonaiOnSagemakerRoleParameter",
            allowed_pattern=".*",
            description="Role for Sagemaker to access various services",
            parameter_name="MonaiOnSagemakerRoleParameter",
            string_value=monaiOnSagemakerRole.role_arn,
            tier=ssm.ParameterTier.STANDARD
        )


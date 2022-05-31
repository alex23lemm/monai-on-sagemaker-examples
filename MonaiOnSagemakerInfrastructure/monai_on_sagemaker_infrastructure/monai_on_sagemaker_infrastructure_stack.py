from aws_cdk import (
    # Duration,
    Stack,
    aws_efs as efs,
    aws_ec2 as ec2,
    aws_ssm as ssm
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
        
        
        sagemaker_monai_efs_id_parameter = ssm.StringParameter(
            self, 
            "SageMakerMonaiEFSIDParameter",
            allowed_pattern=".*",
            description="EFS ID for Sagemaker Monai",
            parameter_name="SageMakerMonaiEFSIDParameter",
            string_value=file_system.file_system_id,
            tier=ssm.ParameterTier.STANDARD
        )

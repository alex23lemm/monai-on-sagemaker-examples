import aws_cdk as core
import aws_cdk.assertions as assertions

from monai_on_sagemaker_infrastructure.monai_on_sagemaker_infrastructure_stack import MonaiOnSagemakerInfrastructureStack

# example tests. To run these tests, uncomment this file along with the example
# resource in monai_on_sagemaker_infrastructure/monai_on_sagemaker_infrastructure_stack.py
def test_sqs_queue_created():
    app = core.App()
    stack = MonaiOnSagemakerInfrastructureStack(app, "monai-on-sagemaker-infrastructure")
    template = assertions.Template.from_stack(stack)

#     template.has_resource_properties("AWS::SQS::Queue", {
#         "VisibilityTimeout": 300
#     })

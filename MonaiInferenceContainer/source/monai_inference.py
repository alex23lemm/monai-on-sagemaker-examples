"""Inference function overrides for SageMaker PyTorch serving container
"""
# Python Built-Ins:
import json
import logging
import sys

# External Dependencies:
import torch

# Local Dependencies:
#from model import MNISTNet

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_fn(model_dir):
    print("*********      Load saved model from file ************** ")
    #model = MNISTNet().to(device)
    #model.eval()
    model = ""
    return model


def input_fn(request_body, request_content_type):
    print(" **************** Validate, de-serialize and pre-process requests **************** ")
    assert request_content_type == "application/json"
    #data = json.loads(request_body)["inputs"]
    #data = torch.tensor(data, dtype=torch.float32, device=device)
    data = ""
    return data


def predict_fn(input_object, model):
    print(" ************************ Execute the model on input data **********************")
    #with torch.no_grad():
    #    prediction = model(input_object)
    prediction = ""
    return prediction


def output_fn(predictions, content_type):
    print(" ************************* Post-process and serialize model output to API response ***************** ")
    assert content_type == "application/json"
    #res = predictions.cpu().numpy().tolist()
    res = " return from dummy prediction"
    return json.dumps(res)
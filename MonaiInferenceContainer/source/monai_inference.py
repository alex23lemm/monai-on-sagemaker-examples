"""Inference function overrides for SageMaker PyTorch serving container
"""
# Python Built-Ins:
import json
import logging
import sys
import argparse
import ast
import os

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        metavar="W",
        help="number of data loading workers (default: 2)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="E",
        help="number of total epochs to run (default: 2)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, metavar="BS", help="batch size (default: 4)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="initial learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, metavar="M", help="momentum (default: 0.9)"
    )
    parser.add_argument(
        "--dist-backend", type=str, default="gloo", help="distributed backend (default: gloo)"
    )

    # The parameters below retrieve their default values from SageMaker environment variables, which are
    # instantiated by the SageMaker containers framework.
    # https://github.com/aws/sagemaker-containers#how-a-script-is-executed-inside-the-container
    parser.add_argument("--hosts", type=str, default=ast.literal_eval(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
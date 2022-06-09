import logging
import json
import torch 
import numpy as np
from six import BytesIO
import io
from PIL import Image
import tempfile
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader, decollate_batch
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureChannelFirstd,
    EnsureTyped,
    EnsureType,
)
from monai.utils import set_determinism

from monai.apps import download_and_extract

import torch

VAL_AMP = True

logger = logging.getLogger(__name__)
def model_fn(model_dir):
    print("Model Dir : ", model_dir)
    
    logger.info("model dir %s", model_dir)
        
    device = get_device()
    print('device is')
    print(device)
    
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=4,
        out_channels=3,
        dropout_prob=0.2,
    )
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)
    model.to(device)
    model.load_state_dict(
        torch.load(model_dir + '/model.pth')
    )
    model.eval()
    #model = torch.load(model_dir + '/model.pth', map_location=torch.device(device))
    print(type(model))
    return model


def input_fn(request_body, request_content_type):
    # frame_width = 1024
    # frame_height = 1024
    # interval = 30
    # f = io.BytesIO(request_body)
    # tfile = tempfile.NamedTemporaryFile(delete=False)
    # tfile.write(f.read())
    # print(tfile.name)
    # video_frames = video2frame(tfile,frame_width, frame_height, interval)  
    # #convert to tensor of float32 type
    # transform = transforms.Compose([
    #     transforms.Lambda(lambda video_frames: torch.stack([transforms.ToTensor()(frame) for frame in video_frames])) # returns        a 4D tensor
    # ])
    # image_tensors = transform(video_frames)
    image_tensors = np.zeros(1)
    return image_tensors

def predict_fn(data, model):
    print('in custom predict function')
    with torch.no_grad():
    #     device = get_device()
    #     model = model.to(device)
    #     input_data = data.to(device)
    #     model.eval()
    #     output = model(input_data)
        dummy_data = torch.tensor(np.zeros((256,256,256)))
        output = inference(dummy_data, model)    
    return output

    
def output_fn(output_batch, accept='application/json'):
    res = []
    print('output list length')
    print(len(output_batch))
    for output in output_batch:
         res.append({'boxes':output['boxes'].detach().cpu().numpy().tolist(),'labels':output['labels'].detach().cpu().numpy().tolist(),'scores':output['scores'].detach().cpu().numpy().tolist()})
    
    return json.dumps(res)

def get_device():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return device

def inference(input, model):
    
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(240, 240, 160),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )

    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)

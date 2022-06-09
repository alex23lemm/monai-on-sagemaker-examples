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

from monai.data import (
    ArrayDataset, GridPatchDataset, create_test_image_3d, PatchIter)
from monai.utils import first

from monai.transforms import (
    AddChannel,
    Compose,
    LoadImage,
    RandSpatialCrop,
    ScaleIntensity,
    EnsureType,
)

import torch

import boto3

import sagemaker.s3 as sagemaker_s3

import tempfile

from zmq import device

VAL_AMP = True

logger = logging.getLogger(__name__)
def model_fn(model_dir):
    print("Model Dir : ", model_dir)
    
    logger.info("model dir %s", model_dir)
        
    device = get_device()
    print('device is : ', device)
    
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
    model.load_state_dict(
        torch.load(model_dir + '/model.pth')
    )
    model.to(device)
    model.eval()
    #model = torch.load(model_dir + '/model.pth', map_location=torch.device(device))
    print("Model Type : ", type(model))
    return model




class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(np.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(
                np.logical_or(
                    np.logical_or(d[key] == 2, d[key] == 3), d[key] == 1
                )
            )
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = np.stack(result, axis=0).astype(np.float32)
        return d




def input_fn(request_body, request_content_type):
    f = io.BytesIO(request_body)
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz")
    tfile.write(f.read())
    tfile.seek(0)
    tfile.close()
    print("Temporary filename: ", tfile.name)
    
    
    
    #s3_data_uri = "s3://bcinspectio/old/brain_tumor/Task01_BrainTumor/imagesTr/BRATS_001.nii.gz"
    #s3_data_uri = request_body.decode()
    #sagemaker_s3.S3Downloader.download(s3_data_uri, "datasets")
    imtrans = Compose(
        [
            LoadImage(image_only=True),
            ScaleIntensity(),
            EnsureType(),
        ]
    )
    val_transform = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"]),
    ]
    )
    #from os import walk
    #images = next(walk("datasets"), (None, None, []))[2]  # [] if no file
    #for i in range(len(images)):
    #    images[i] = "datasets/" + images[i]
    #print(images)
    images = []
    images.append(tfile.name)
    print("Images list :", images)
    ds = ArrayDataset(images,imtrans)
    print("Monai Dataset : ", ds)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, num_workers=0, pin_memory=torch.cuda.is_available()
    )
    print("Loader : ", loader)
    im = first(loader)
    device = get_device()
    print('device is : ', device)
    im.to(device)
    print("Image shape : ", im.shape)
    return im

def predict_fn(data, model):
    print(' ********  custom predict function *******')
    image_tensor = data
    print("Input Tesor size : ", image_tensor.shape)
    print("Model Type : ", type(model))
    with torch.no_grad():
        temp = []
        temp.append(image_tensor[0,:,:,:,0])
        temp.append(image_tensor[0,:,:,:,1])
        temp.append(image_tensor[0,:,:,:,2])
        temp.append(image_tensor[0,:,:,:,3])

        inf_data = torch.stack(temp)
        inf_data = inf_data.expand(1,inf_data.shape[0],inf_data.shape[1], inf_data.shape[2], inf_data.shape[3] )
        print("Inference Data Tensor Size : ", inf_data.shape)
        output = inference(inf_data, model)
        print("Prediction Tensor Size : ", output.shape)    
    return output

    
def output_fn(output_batch, accept='application/octet-stream'):
    
    print("output : ", output_batch)
    
    return output_batch
    # res = []
    # print('output list length')
    # print(len(output_batch))
    # for output in output_batch:
    #      res.append({'boxes':output['boxes'].detach().cpu().numpy().tolist(),'labels':output['labels'].detach().cpu().numpy().tolist(),'scores':output['scores'].detach().cpu().numpy().tolist()})
    
    # return json.dumps(res)

def get_device():
    #device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = "cpu"
    return device

def inference(input, model):
    
    def _compute(input):
        print("Monai Model inside Compute : ")
        print("Model Type : ", type(model))
        print("Sliding Window Inference Object : ", sliding_window_inference)
        try:
            device = get_device()
            print("device: ", device)
            input.to(device)
            model.to(device)
            slid_window_inference = sliding_window_inference(
                inputs=input,
                roi_size=(240, 240, 160),
                sw_batch_size=1,
                predictor=model,
                overlap=0.5,
            )
        except Exception as e:
            print("Exception During Inference: ", e)
        print("Sliding Window Inference Result Shape : ", slid_window_inference.shape)
        return slid_window_inference

    if VAL_AMP:
        with torch.cuda.amp.autocast():
            print("VAL AMP Custom Monai Inference")
            print("Custom Inference Input Shape : ", input.shape)
            inf_result = _compute(input)
            print("Inference result shape : ", inf_result.shape)
            return inf_result
    else:
        print("Custom Monai Inference")
        return _compute(input)

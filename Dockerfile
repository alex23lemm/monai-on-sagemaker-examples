
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.11.0-gpu-py38-cu113-ubuntu20.04-sagemaker
RUN apt-get update 
RUN apt-get install -y git
RUN pip install --upgrade pip
RUN pip install ipykernel && \
    python -m ipykernel install --sys-prefix && \
    pip install --no-cache-dir \
    'monai[gdown, nibabel, tqdm, ignite]' \
    'boto3' \
    'matplotlib' \
    'jupyter' \
    'ipywidgets' \
    'widgetsnbextension'

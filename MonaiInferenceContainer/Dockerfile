#FROM projectmonai/monai



FROM 763104351884.dkr.ecr.eu-west-1.amazonaws.com/pytorch-inference:1.11.0-gpu-py38-cu113-ubuntu20.04-sagemaker
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

COPY monai_inference.py monai_inference.py 

ENTRYPOINT ["python", "monai_inference.py"]
FROM 763104351884.dkr.ecr.eu-west-1.amazonaws.com/pytorch-training:1.11.0-gpu-py38-cu113-ubuntu20.04-sagemaker
#FROM 763104351884.dkr.ecr.eu-west-1.amazonaws.com/pytorch-inference:1.11.0-gpu-py38-cu113-ubuntu20.04-sagemaker


RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python3-pip \
         python3-setuptools \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

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
    'widgetsnbextension' \
    'flask' \
    'gunicorn' \
    'causalnex'

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY source /opt/program
WORKDIR /opt/program

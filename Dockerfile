FROM projectmonai/monai
RUN apt-get update 
RUN apt-get install -y git
RUN pip install --upgrade pip
RUN pip install ipykernel && \
    python -m ipykernel install --sys-prefix && \
    pip install --no-cache-dir \
    'boto3' \
    'sagemaker' \
    'matplotlib' \
    'jupyter' \
    'ipywidgets' \
    'widgetsnbextension'

COPY train.py /root/train.py 
FROM supervisely/base-py-sdk:6.35.0
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update

RUN pip3 install albumentations==1.1.0
RUN pip3 install timm==0.4.12 
RUN pip3 install numpy==1.19.1
RUN pip3 install opencv_python_headless==4.4.0.44
RUN pip3 install pytorch_lightning==0.9.0
RUN pip3 install transformers==3.3.1
RUN pip3 install scipy==1.5.2
RUN pip3 install apex==0.9.10dev
RUN pip3 install scikit_learn==0.23.2
RUN pip3 install python-dotenv==0.19.2
RUN pip3 install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

RUN apt-get -y install curl
RUN apt -y install wireguard iproute2
RUN apt-get -y install wget
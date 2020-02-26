from ubuntu:18.04
LABEL maintainer="Ivan Puzako"
RUN apt-get update && yes | apt-get upgrade
RUN apt-get install -y python3
RUN apt-get install -y git python3-pip 
WORKDIR /home/detection
RUN pip3 install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install opencv-python 
RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN git clone https://github.com/Ivanpuzako/mleng-task.git



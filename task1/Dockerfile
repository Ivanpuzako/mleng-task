from ubuntu:18.04
LABEL maintainer="Ivan Puzako"
WORKDIR /home/detection
RUN apt-get update && yes | apt-get upgrade && \
    apt-get install -y python3 && \
    apt-get install -y git python3-pip && \ 
    pip3 install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
    pip3 install opencv-python && \
    apt-get update && \
    apt-get install -y libsm6 libxext6 libxrender-dev && \
    git clone https://github.com/Ivanpuzako/mleng-task.git && \
    addgroup --gid 1024 mygroup && \
    adduser --disabled-password --gecos "" --force-badname --gid 1024 myuser
USER myuser
 


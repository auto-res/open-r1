
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install required system packages and Python
RUN apt-get update && \
apt-get install -y python3.10 python3-pip git tmux curl net-tools git-lfs && \
apt-get clean && \
rm -rf /var/lib/apt/lists/*

RUN curl -s -O https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh && \
    bash elan-init.sh -y && \
    rm -rf elan-init.sh

ENV PATH="/root/.elan/bin:${PATH}"

RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Set default work directory back to project root
WORKDIR /app
RUN pip install uv

CMD ["/bin/bash"]
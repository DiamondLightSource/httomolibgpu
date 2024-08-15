FROM nvidia/cuda:11.6.2-devel-ubi8

RUN dnf install -y git && \
    dnf clean all

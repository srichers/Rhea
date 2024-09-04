FROM nvidia/cuda:11.4.3-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3 python3-pip gfortran build-essential libhdf5-openmpi-dev openmpi-bin pkg-config libopenmpi-dev openmpi-bin libblas-dev liblapack-dev libpnetcdf-dev git python-is-python3 gnuplot cmake wget
RUN pip3 install numpy matplotlib h5py scipy sympy yt torch
RUN wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu118.zip && unzip libtorch-cxx11-abi-shared-with-deps-2.0.1+cu118.zip && mv libtorch /usr/local/libtorch
ENV USER=jenkins
ENV LOGNAME=jenkins

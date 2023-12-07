#!/bin/bash
export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_RANK 
python latency_mnist_mpi.py 

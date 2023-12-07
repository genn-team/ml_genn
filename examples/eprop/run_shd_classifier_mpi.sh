#!/bin/bash
export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_RANK 
python shd_classifier_mpi.py 

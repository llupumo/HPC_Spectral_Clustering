#!/bin/bash

rsync -av --exclude-from='/home/llu/Programs/HPC_Spectral_Clustering/rsync_exclude.txt' --exclude=/demos/submit_job.sh llpui9007@fram.sigma2.no:/cluster/home/llpui9007/Programs/HPC_Spectral_Clustering/ /home/llu/Programs/HPC_Spectral_Clustering/

# !/bin/bash

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <executable> <number of processes> <number of threads>" >&2
  exit 1
fi

export DAPL_DBG_TYPE=0

DATAPATH=/home/2023-spring/data/stencil_data

export OMP_NUM_THREADS=$3
export OMP_PLACES=cores
export OMP_PROC_BIND=close
srun -N 1 -n $2 --cpus-per-task=$3 ./$1 27 256 256 256 16  ${DATAPATH}/stencil_data_256x256x256 ${DATAPATH}/stencil_answer_27_256x256x256_16steps
#srun -N 1 -n $2 ./$1 27 384 384 384 16 ${DATAPATH}/stencil_data_384x384x384 ${DATAPATH}/stencil_answer_27_384x384x384_16steps
#srun -N 1 -n $2 ./$1 27 512 512 512 16 ${DATAPATH}/stencil_data_512x512x512 ${DATAPATH}/stencil_answer_27_512x512x512_16steps

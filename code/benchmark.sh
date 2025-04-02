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
srun -N 1 -n $2 --cpus-per-task=$3 ./$1 27 256 256 256 100 ${DATAPATH}/stencil_data_256x256x256
#salloc -N $2 --ntasks-per-node $3 mpirun $1 7  100 ${DATAPATH}/stencil_data_384x384x384
#salloc -N $2 --ntasks-per-node $3 mpirun $1 7 512 512 512 100 ${DATAPATH}/stencil_data_512x512x512
#salloc -N $2 --ntasks-per-node $3 mpirun $1 27 256 256 256 100 ${DATAPATH}/stencil_data_256x256x256
#salloc -N $2 --ntasks-per-node $3 mpirun $1 27 384 384 384 100 ${DATAPATH}/stencil_data_384x384x384
#salloc -N $2 --ntasks-per-node $3 mpirun $1 27 512 512 512 100 ${DATAPATH}/stencil_data_512x512x512
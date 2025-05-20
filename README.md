- Data generation: `python3 generate-data.py --num_systems <number-of-systems> --system_size <size-of-system(s)>`
- Source code: can change data path and recompile
- Available executable files: compiled to run data with corresponding data size, can run data generation code for new data
+ For sequential algorithm: `./seq-<system-size>`
+ For parallel algorithm: `mpirun -np <num-processes> ./mpi-<system-size>`
# A Parallel Implementation of Gauss-Seidel Algorithm using MPI
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Running the Program](#running-the-program)

## Introduction
This is the Capstone Project of Group 8 for the course Parallel and Distributed Programming. Our topic is on the Gauss-Seidel method for iteratively solving a linear system of equations. We will be providing a sequential implementation of the method and a parallel one using Message Passing Interface (MPI) standard.

## Prerequisites
- This project is carried out entirely in C, so you should have an environment capable of running C programs. For following sections, we will be using `gcc` compiler; you can change this to the compiler that you use in commands below.
- We use MPICH, an implementation of MPI standard, for our parallel algorithm. You should have MPICH installed to your device so that you can compile and run the parallel code.
- For data generation, we use a Python code with the NumPy library for random number generation. If you do not have Python available, you can use available data samples.

## Running the Program

### Data Generation
The algorithm is only guaranteed to converge under certain conditions of the linear system. We provide available data samples in selected sizes, but you can run the following command to create data with different sizes or to renew existing samples:
```bash
python3 generate-data.py \
--num_systems <number-of-systems> \
--system_size <size-of-systems>
```
Provide your desired number of systems and their size; default is 1 system with 50 equations and variables. Data is saved to the `data` directory, with name of files in the format `system_<index-of-system>_size<size-of-system>`.

### Compile and Run for new data
To compile the code for new data, change the `FILENAME` variable in the source code to the name of data file you want to use.

- For sequential algorithm:
```bash
gcc gauss-seidel-seq.c
./a.out
```

- For parallel algorithm:
```bash
mpicc gauss-seidel-mpi.c
mpirun -np <num-of-processes> ./a.out
```

### Available Executable Files
We provide several executable files compiled to run on data with corresponding data size. Simply replace `<system-size>` with your desired size from the following list: `5, 50, 100, 500, 1000, 2000`

- For sequential algorithm: 
```bash
./seq-<system-size>
```

- For parallel algorithm: 
```bash
mpirun -np <num-processes> ./mpi-<system-size>
```
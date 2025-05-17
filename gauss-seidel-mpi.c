#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

// #define M 2
#define N 5
// #define P 1
#define ERR 0.000001
#define MAX_ITER 100

/* 
    use element-based formula of Gauss-Seidel algorithm
    Ax = b; A_MxN, b_Nx1, x_Nx1
    x_k+1[i] = 1/A[i,i] * (b[i] - <sum(a[i,j] * x_k+1[j]) with j=1 to i-1> - <sum(a[i,j] * x_k[j]) with j=i+1 to N>)

    adjust for parallelism: only use current iteration data within domain, use prev. iter data even for solutions at prior positions
*/
void display_vector(float* A, int size) {
    for (int i = 0; i < size; i++) {
        printf("%.3f\t", *(A+i));
    }
}
void display_matrix(float* A, int row, int col) {
    for (int i = 0; i < row; i++) {
        display_vector(A+i*col, col);
        printf("\n");
    }
}

int main(int argc, char* argv[]) {
    int rank, num_p;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_p);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // MPI_Status status;
    float *A, *b, *x, *xs;
    int i,j,domain_len, start, stop;

    domain_len = N / num_p; //handles non-divisibility
    start = rank*domain_len; stop = start+domain_len;

    A = (float*)malloc(N*N*sizeof(float));
    x = (float*)malloc(N*sizeof(float));
    b = (float*)malloc(N*sizeof(float));
    xs = (float*)malloc(domain_len*sizeof(float));

    if (rank == 0) {
        // initialize data
        srand(time(0));
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++)
                *(A + j*N + i) = j*N+i+rand()%100+1;
            *(x + i) = 0;
            *(b + i) = i+rand()%100+1;
        }
        // A[0]=16;A[1]=3;A[2]=7;A[3]=-11;
        // b[0]=11;b[1]=13; //x[0]=3,x[1]=1
        // x[0]=1;x[1]=1;

        printf("Matrix A:\n");
        display_matrix(A, N, N);
        printf("Vector b:\n");
        display_vector(b, N);
        printf("\nInitial Solution:\n");
        display_vector(x,N);
    }
    // broadcast data; try scattering data since each process only use <domain_len> rows of A and entries of B
    MPI_Bcast(A, N*N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(b, N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    // float *As, *bs;
    // As = (float*)malloc(domain_len*N*sizeof(float));
    // bs = (float*)malloc(domain_len*sizeof(float));
    // MPI_Scatter(A, domain_len*N, MPI_FLOAT, As, domain_len*N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    // MPI_Scatter(b, domain_len, MPI_FLOAT, bs, domain_len, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(x, N, MPI_FLOAT, 0, MPI_COMM_WORLD); // all processes hold entire vector of previous iter
    MPI_Scatter(x, domain_len, MPI_FLOAT, xs, domain_len, MPI_FLOAT, 0, MPI_COMM_WORLD); // each process calculates on its own domain
    
    // measure runtime
    clock_t begin, end;
    if (rank == 0) begin = clock();

    int iter;
    float max_err_local, max_err_global, sum_before, sum_between, sum_after, new_solution, cur_err;
    for (iter = 1; iter <= MAX_ITER; iter++) {
        max_err_local = 0;
        for (i = start; i < stop; i++) {
            sum_before = sum_between = sum_after = 0;

            for (j = 0; j < start; j++) 
                sum_before += (*(A+i*N+j) * *(x+j)); // calculate before domain using prev. iter solution
            for (j = i+1; j < N; j++) 
                sum_after += (*(A+i*N+j) * *(x+j)); // calculate after current position using prev. iter solution
            for (j = start; j < i; j++)
                sum_between += (*(A+i*N+j) * *(xs+j-start)); // calculate in domain up to before current position using current iter. solution
            new_solution = (*(b+i) - sum_before - sum_between - sum_after) / *(A+i*N+i);
            
            cur_err = fabs(new_solution - *(x+i));
            if (cur_err > max_err_local) max_err_local = cur_err;
            *(xs+i-start) = new_solution;
        }

        // gather solutions and broadcast to all processes
        MPI_Allgather(xs, domain_len, MPI_FLOAT, x, domain_len, MPI_FLOAT, MPI_COMM_WORLD);

        // collect local max, obtain global max and return to all processes; check for stopping condition
        MPI_Allreduce(&max_err_local, &max_err_global, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
        if (max_err_global <= ERR) break;
    }

    if (rank == 0) {
        end = clock();
        printf("\nApproximate solution after %d iterations:\n", iter);
        display_vector(x,N);
        double run_time = (double) (end-begin)/CLOCKS_PER_SEC;
        printf("\nTotal running time: %f", run_time);
    }
    free(A);
    free(b);
    free(x);
    // free(As);
    // free(bs);
    free(xs);
    MPI_Finalize();
}
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

// #define N 5
#define ERR 0.000001
#define MAX_ITER 100
#define FILENAME "data/system_4_size100.txt"

/* 
    use element-based formula of Gauss-Seidel algorithm
    Ax = b; A_MxN, b_Nx1, x_Nx1
    x_k+1[i] = 1/A[i,i] * (b[i] - <sum(a[i,j] * x_k+1[j]) with j=1 to i-1> - <sum(a[i,j] * x_k[j]) with j=i+1 to N>)

    adjust for parallelism: only use current iteration data within domain, 
    use prev. iter data even for solutions at prior positions
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
int read_data(int *size, float **A, float **b, float **solution) {
    FILE *f = fopen(FILENAME, "r");
    if (!f) {
        printf("Failed to read input data!");
        return 0;
    }

    fscanf(f, "%d", size); // read size of matrix
    int N = *size;
    *A = (float*)malloc(N*N*sizeof(float));
    *solution = (float*)malloc(N*sizeof(float));
    *b = (float*)malloc(N*sizeof(float));
    for (int i = 0; i < *size; i++) {
        for (int j = 0; j < *size; j++) {
            fscanf(f, "%f", *A+i**size+j);
        }
    }
    for (int i = 0; i < *size; i++) fscanf(f, "%f", *b+i);
    for (int i = 0; i < *size; i++) fscanf(f, "%f", *solution+i);

    fclose(f);
    return 1;
}

int main(int argc, char* argv[]) {
    int rank, num_p;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_p);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status status;
    float *A, *b, *solution, *x, *As, *bs;
    int N, i, j, domain_len, remainder, start, stop, *domain_lens, *offsets, *domain_lens_A, *offsets_A;
    short tag_send_A = 42, tag_send_b = 44;

    if (rank == 0) {
        // Read data
        int input_status = read_data(&N, &A, &b, &solution);
        if (!input_status) {
            MPI_Finalize();
            return 0;
        }
        domain_lens = (int*)malloc(num_p*sizeof(int));
        offsets = (int*)malloc(num_p*sizeof(int));
        domain_lens_A = (int*)malloc(num_p*sizeof(int));
        offsets_A = (int*)malloc(num_p*sizeof(int));

        // printf("Matrix A:\n");
        // display_matrix(A, N, N);

        // printf("Vector b:\n");
        // display_vector(b, N);

        // printf("Solution:\n");
        // display_vector(solution, N);

        // printf("\nInitial Solution:\n");
        // display_vector(x,N);
    }
    // broadcast problem size N and initial solution x
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // define domain for each process
    domain_len = N / num_p;
    remainder = N % num_p;
    if (rank < remainder) {
        start = rank * domain_len + rank;
        stop = start + domain_len + 1;
    }
    else {
        start = rank * domain_len + remainder;
        stop = start + domain_len;
    }

    x = (float*)malloc(N*sizeof(float));
    // xs = (float*)malloc((stop-start)*sizeof(float));
    if (rank == 0) {
        for (i = 0; i < N; i++) {
            *(x + i) = 0;
        }
        for (i = 0; i < num_p; i++) {
            if (i < remainder) {
                *(domain_lens + i) = domain_len + 1;
                *(offsets + i) = i * domain_len + i;
                *(domain_lens_A + i) = (domain_len+1) * N;
                *(offsets_A + i) = i * (domain_len+1) * N;
            }
            else {
                *(domain_lens + i) = domain_len;
                *(offsets + i) = i * domain_len + remainder;
                *(domain_lens_A + i) = domain_len * N;
                *(offsets_A + i) = i * domain_len * N + remainder * N;
            }
        }
    }
    MPI_Bcast(x, N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    // MPI_Scatterv(x, domain_lens, offsets, MPI_FLOAT, xs, domain_len+1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // broadcast data; try scattering data since each process only use <domain_len> rows of A and entries of B
    As = (float*)malloc((stop-start)*N * sizeof(float));
    bs = (float*)malloc((stop-start) * sizeof(float));

    // Failed: sending to itself will block indefinitely
    // if (rank == 0) {
    //     for (i = 0; i < num_p; i++) {
    //         if (i < remainder) {
    //             MPI_Send(A+i*(domain_len+1)*N, (domain_len+1)*N, MPI_FLOAT, i, tag_send_A, MPI_COMM_WORLD);
    //             MPI_Send(b+i*(domain_len+1), domain_len+1, MPI_FLOAT, i, tag_send_b, MPI_COMM_WORLD);
    //         }
    //         else {
    //             MPI_Send(A+i*domain_len*N+remainder*N, domain_len*N, MPI_FLOAT, i, tag_send_A, MPI_COMM_WORLD);
    //             MPI_Send(b+i*domain_len+remainder, domain_len, MPI_FLOAT, i, tag_send_b, MPI_COMM_WORLD);
    //         }
    //     }
    // }
    // MPI_Recv(As, (stop-start)*N, MPI_FLOAT, 0, tag_send_A, MPI_COMM_WORLD, &status);
    // MPI_Recv(bs, stop-start, MPI_FLOAT, 0, tag_send_b, MPI_COMM_WORLD, &status);

    MPI_Scatterv(A, domain_lens_A, offsets_A, MPI_FLOAT, As, (domain_len+1)*N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(b, domain_lens, offsets, MPI_FLOAT, bs, domain_len+1, MPI_FLOAT, 0, MPI_COMM_WORLD);


    // measure runtime
    clock_t begin, end;
    int iter;
    float cur_err, max_err_local, max_err_global, sum_before, sum_after, new_solution;
    if (rank == 0) begin = clock();
    for (iter = 1; iter <= MAX_ITER; iter++) {
        max_err_local = 0;
        for (i = start; i < stop; i++) {
            sum_before = sum_after = 0;

            for (j = 0; j < i; j++) 
                sum_before += (*(As+(i-start)*N+j) * *(x+j)); // calculate before domain using prev. iter solution
            for (j = i+1; j < N; j++) 
                sum_after += (*(As+(i-start)*N+j) * *(x+j)); // calculate after current position using prev. iter solution
            // for (j = start; j < i; j++)
            //     sum_between += (*(As+(i-start)*N+j) * *(xs+j-start));
            new_solution = (*(bs+i-start) - sum_before - sum_after) / *(As+(i-start)*N+i);
            
            cur_err = fabs(*(x+i) - new_solution);
            if (cur_err > max_err_local) max_err_local = cur_err;
            *(x+i) = new_solution;
        }

        // gather solutions and broadcast to all processes
        for (j = 0; j < num_p; j++) {
            if (j < remainder) MPI_Bcast(x+j*(domain_len+1), domain_len+1, MPI_FLOAT, j, MPI_COMM_WORLD);
            else MPI_Bcast(x+j*domain_len+remainder, domain_len, MPI_FLOAT, j, MPI_COMM_WORLD);
        }
        // MPI_Allgatherv(xs, domain_len+1, MPI_FLOAT, x, domain_lens, offsets, MPI_FLOAT, MPI_COMM_WORLD);
        // !!! need xs for gathering

        // collect local max, obtain global max and return to all processes; check for stopping condition
        MPI_Allreduce(&max_err_local, &max_err_global, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
        if (max_err_global <= ERR) break;
    }

    if (rank == 0) {
        end = clock();
        // printf("\nApproximate solution after %d iterations:\n", iter);
        // display_vector(x,N);

        float diff = 0;
        for (i = 0; i < N; i++) {
            diff += fabs(*(x+i) - *(solution+i));
        }
        diff /= N;

        if (iter <= MAX_ITER) {
            printf("\nAlgorithm converged after %d iterations.", iter);
        } else {
            printf("\nAlgorithm failed to converge after %d iterations.", MAX_ITER);
        }
        if (diff < ERR) {
            printf("\nApproximation accepted with average error < %.6f.", ERR);
        } else {
            printf("\nApproximation not accpeted with average error = %.6f", diff);
        }

        double run_time = (double) (end-begin)/CLOCKS_PER_SEC;
        printf("\nTotal running time: %f", run_time);
        free(A);
        free(b);
        free(solution);
        free(domain_lens); free(domain_lens_A);
        free(offsets); free(offsets_A);
    }
    free(x);
    free(As);
    free(bs);
    // free(xs);
    
    MPI_Finalize();
    return 0;
}
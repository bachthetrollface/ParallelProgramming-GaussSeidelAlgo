#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// #define N 5
#define ERR 0.000001
#define MAX_ITER 100
#define FILENAME "data/system_2_size5.txt"

/* 
    use element-based formula of Gauss-Seidel algorithm
    Ax = b; A_MxN, b_Nx1, x_Nx1
    x_k+1[i] = 1/A[i,i] * (b[i] - <sum(a[i,j] * x_k+1[j]) with j=1 to i-1> - <sum(a[i,j] * x_k[j]) with j=i+1 to N>)
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

int main() {
    float *A, *b, *x, *solution;
    int i,j,N;

    int input_status = read_data(&N, &A, &b, &solution);
    if (!input_status) return 0;
    x = (float*)malloc(N*sizeof(float));
    for (i = 0; i < N; i++) {
        *(x+i) = 0;
    }

    // printf("Matrix A:\n");
    // display_matrix(A, N, N);
    // printf("Vector b:\n");
    // display_vector(b, N);
    // printf("\nSolution:\n");
    // display_vector(solution,N);
    // printf("\nInitial Solution:\n");
    // display_vector(x,N);

    clock_t begin, end;
    begin = clock();
    int iter = 0;
    float max_err = 999, sum_before, sum_after, tmp, cur_err;
    for (iter = 1; iter <= MAX_ITER; iter++) {
        max_err = 0;
        for (i = 0; i < N; i++) {
            // sum_before = sum_after = 0;
            // for (j = 0; j < i; j++) sum_before += (*(A+i*N+j) * *(x+j));
            // for (j = i+1; j < N; j++) sum_after += (*(A+i*N+j) * *(x+j));
            // tmp = (*(b+i) - sum_before - sum_after) / *(A+i*N+i);

            tmp = *(b+i);
            for (j = 0; j < N; j++) if (j != i) tmp -= (*(A+i*N+j) * *(x+j));
            tmp /= *(A+i*N+i);
            
            cur_err = fabs(tmp - *(x+i));
            if (cur_err > max_err) max_err = cur_err;
            *(x+i) = tmp;
        }
        if (max_err <= ERR) break;
    }
    end = clock();
    double run_time = (double) (end-begin)/CLOCKS_PER_SEC;

    float diff = 0;
    for (i = 0; i < N; i++) {
        diff += fabs(*(x+i) - *(solution+i));
    }
    diff /= N;
    
    printf("\nApproximate solution after %d iterations:\n", iter);
    display_vector(x,N);
    if (diff < ERR) {
        printf("\nApproximation accepted with average error < %.6f.", ERR);
    } else {
        printf("\nApproximation not accpeted with average error = %.6f", diff);
    }
    printf("\nTotal running time: %f", run_time);
    free(A);
    free(b);
    free(x);
}
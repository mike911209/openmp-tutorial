/*
**  PROGRAM: jacobi Solver
**
**  PURPOSE: This program will explore use of a jacobi iterative
**           method to solve a system of linear equations (Ax= b).
**
**           Here is the basic idea behind the method.   Rewrite
**           the matrix A as a Lower Triangular (L), upper triangular
**           (U) and diagonal matrix (D)
**
**                Ax = (L + D + U)x = b
**
**            Carry out the multiplication and rearrange:
**
**                Dx = b - (L+U)x  -->   x = (b-(L+U)x)/D
**
**           We can do this iteratively
**
**                x_new = (b-(L+U)x_old)/D
**
**  USAGE:   Run wtihout arguments to use default SIZE.
**
**              ./jac_solv
**
**           Run with a single argument for the order of the A
**           matrix ... for example
**
**              ./jac_solv 2500
**
**  HISTORY: Written by Tim Mattson, Oct 2015
*/

#include <random>

// #include "mm_utils.h" //a library of basic matrix utilities functions
// and some key constants used in this program
//(such as TYPE)

using TYPE = double;

std::mt19937 rng(std::random_device{}());
std::uniform_int_distribution<int> uid(0, RAND_MAX);

void init_diag_dom_near_identity_matrix(const int Ndim, TYPE *const A)
{
    // Create a random, diagonally dominant matrix.  For
    // a diagonally dominant matrix, the diagonal element
    // of each row is great than the sum of the other
    // elements in the row.  Then scale the matrix so the
    // result is near the identiy matrix.
    for (int i = 0; i < Ndim; i++)
    {
        TYPE sum = 0.0;
        for (int j = 0; j < Ndim; j++)
            sum += *(A + i * Ndim + j) = (uid(rng) % 23) / 1000.0;
        *(A + i * Ndim + i) += sum;

        // scale the row so the final matrix is almost an identity matrix;wq
        for (int j = 0; j < Ndim; j++)
            *(A + i * Ndim + j) /= sum;
    }
}

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

#include <cuda.h>

constexpr double TOLERANCE = 0.001,
                 LARGE = 1000000.0;
constexpr int DEF_SIZE = 1024,
              MAX_ITERS = 65536;

// #define DEBUG    1     // output a small subset of intermediate values
// #define VERBOSE  1

__global__ void jacobi(const unsigned Ndim, TYPE *A, TYPE *b, TYPE *xold, TYPE *xnew) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  xnew[i] = 0.0;
  for (int j = 0; j < Ndim; j++)
    xnew[i] += A[i * Ndim + j] * xold[j] * (i != j);
  xnew[i] = (b[i] - xnew[i]) / A[i * Ndim + i];
}

__global__ void convergence(TYPE *xold, TYPE *xnew, TYPE *conv) {
  extern __shared__ TYPE conv_loc[];
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  TYPE tmp = xnew[i] - xold[i];
  conv_loc[threadIdx.x] = tmp * tmp;

  __syncthreads();

  for (int offset = blockDim.x >> 1; offset; offset >>= 1) {
    if (threadIdx.x < offset)
      conv_loc[threadIdx.x] += conv_loc[threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    conv[threadIdx.x] = conv_loc[0];
}

int main(int argc, char **argv)
{
    // set matrix dimensions and allocate memory for matrices
    const int Ndim = argc > 1 ? atoi(argv[1]) : DEF_SIZE, // A[Ndim][Ndim]
              wgsize = argc > 2 ? atoi(argv[2]) : 64,
              conv_wgsize = argc > 2 ? atoi(argv[2]) : 64,
              dev_idx = argc > 3 ? atoi(argv[3]) : 0;
    TYPE err, chksum,
         *const A = new TYPE[Ndim * Ndim],
         *const b = new TYPE[Ndim],
         *xnew = new TYPE[Ndim],
         *xold = new TYPE[Ndim],
         *const conv_temp = new TYPE[Ndim / conv_wgsize],
         *d_A, *d_b, *d_xnew, *d_xold, *d_conv;

    std::cout << " ndim = " << Ndim << '\n';

    if (!A || !b || !xold || !xnew)
    {
        std::cout << "\n memory allocation error\n";
        exit(-1);
    }

    // generate our diagonally dominant matrix, A
    init_diag_dom_near_identity_matrix(Ndim, A);

    //
    // Initialize x and just give b some non-zero random values
    //
    for (int i = 0; i < Ndim; i++)
    {
        xnew[i] = 0.0;
        xold[i] = 0.0;
        b[i] = (uid(rng) % 51) / 100.0;
    }

    cudaSetDevice(dev_idx);
    
    cudaMalloc(&d_A, sizeof(TYPE) * Ndim * Ndim);
    cudaMalloc(&d_b, sizeof(TYPE) * Ndim);
    cudaMalloc(&d_xnew, sizeof(TYPE) * Ndim);
    cudaMalloc(&d_xold, sizeof(TYPE) * Ndim);
    cudaMalloc(&d_conv, sizeof(TYPE) * (Ndim / conv_wgsize));

    cudaMemcpy(d_A, A, sizeof(TYPE) * Ndim * Ndim, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(TYPE) * Ndim, cudaMemcpyHostToDevice);
    cudaMemcpy(d_xnew, b, sizeof(TYPE) * Ndim, cudaMemcpyHostToDevice);
    cudaMemcpy(d_xold, b, sizeof(TYPE) * Ndim, cudaMemcpyHostToDevice);

    const auto start_time = std::chrono::steady_clock::now();
    //
    // jacobi iterative solver
    //
    TYPE conv = LARGE;
    int iters = 0;
    while (conv > TOLERANCE && iters < MAX_ITERS)
    {
        iters++;

        jacobi<<<Ndim / wgsize, wgsize>>>(Ndim, d_A, d_b, d_xold, d_xnew);
        //
        // test convergence
        //
        convergence<<<Ndim / conv_wgsize, conv_wgsize, sizeof(TYPE) * conv_wgsize>>>(d_xnew, d_xold, d_conv);
        cudaMemcpy(conv_temp, d_conv, sizeof(TYPE) * (Ndim / conv_wgsize), cudaMemcpyDeviceToHost);
        conv = sqrt(std::accumulate(conv_temp, conv_temp + Ndim / conv_wgsize, 0.));

        std::swap(d_xnew, d_xold);
    }
    const std::chrono::duration<double> elapsed_time = std::chrono::steady_clock::now() - start_time;
    std::cout << " Convergence = " << conv << " with " << iters << " iterations and " << elapsed_time.count() << " seconds\n";

    //
    // test answer by multiplying my computed value of x by
    // the input A matrix and comparing the result with the
    // input b vector.
    //
    cudaMemcpy(xnew, d_xnew, sizeof(TYPE) * Ndim, cudaMemcpyDeviceToHost);
    cudaMemcpy(xold, d_xold, sizeof(TYPE) * Ndim, cudaMemcpyDeviceToHost);
    err = 0.0;
    chksum = 0.0;

    for (int i = 0; i < Ndim; i++)
    {
        xold[i] = 0.0;
        for (int j = 0; j < Ndim; j++)
            xold[i] += A[i * Ndim + j] * xnew[j];
        TYPE tmp = xold[i] - b[i];
        chksum += xnew[i];
        err += tmp * tmp;
    }
    err = sqrt(err);
    std::cout << "jacobi solver: err = " << err << ", solution checksum = " << chksum << '\n';
    if (err > TOLERANCE)
        std::cout << "\nWARNING: final solution error > " << TOLERANCE << '\n';

    delete[] A;
    delete[] b;
    delete[] xold;
    delete[] xnew;
    delete[] conv_temp;
}

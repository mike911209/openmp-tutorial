/*
 **  PROGRAM: jacobi Solver ... parallelized with target regions
 **           and (Work in progress) a data region to manage data movement
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
 **           Parallelized by Tim Mattson, Nov 2015
 **           Cleanup by Matt Martineau, May 2018
 */

#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
// and some key constants used in this program
//(such as TYPE)

#define TOLERANCE 0.001
#define DEF_SIZE 1000
#define MAX_ITERS 100000
#define LARGE 1000000.0
#define TYPE double

void init_diag_dom_near_identity_matrix(int Ndim, TYPE *A) {

  int i, j;
  TYPE sum;

  //
  // Create a random, diagonally dominant matrix.  For
  // a diagonally dominant matrix, the diagonal element
  // of each row is great than the sum of the other
  // elements in the row.  Then scale the matrix so the
  // result is near the identiy matrix.
  for (i = 0; i < Ndim; i++) {
    sum = (TYPE)0.0;
    for (j = 0; j < Ndim; j++) {
      *(A + i * Ndim + j) = (rand() % 23) / (TYPE)1000.0;
      sum += *(A + i * Ndim + j);
    }
    *(A + i * Ndim + i) += sum;

    // scale the row so the final matrix is almost an identity matrix;wq
    for (j = 0; j < Ndim; j++)
      *(A + i * Ndim + j) /= sum;
  }
}

// #define DEBUG    1     // output a small subset of intermediate values
//#define VERBOSE  1
#define DEVICE 1

int main(int argc, char **argv) {
  int Ndim; // A[Ndim][Ndim]
  int iters;
  double start_time, elapsed_time;
  TYPE conv, err, chksum;
  TYPE *A, *b, *xold, *xnew;

  // set matrix dimensions and allocate memory for matrices
  if (argc == 2) {
    Ndim = atoi(argv[1]);
  } else {
    Ndim = DEF_SIZE;
  }

  printf(" \n\nJacobi solver, target and data regions ndim = %d\n", Ndim);

#ifdef DEVICE
  int num_devices = omp_get_num_devices();
  int default_device = omp_get_default_device(); // Default device ID

  printf("Number of OpenMP target devices: %d\n", num_devices);
  printf("Default OpenMP target device: %d\n", default_device);
#endif

  A = (TYPE *)malloc(Ndim * Ndim * sizeof(TYPE));
  b = (TYPE *)malloc(Ndim * sizeof(TYPE));
  xold = (TYPE *)malloc(Ndim * sizeof(TYPE));
  xnew = (TYPE *)malloc(Ndim * sizeof(TYPE));

  if (!A || !b || !xold || !xnew) {
    printf("\n memory allocation error\n");
    exit(-1);
  }

  // generate our diagonally dominant matrix, A
  init_diag_dom_near_identity_matrix(Ndim, A);

#ifdef VERBOSE
  mm_print(Ndim, Ndim, A);
#endif

  //
  // Initialize x and just give b some non-zero random values
  //
  for (int i = 0; i < Ndim; i++) {
    xold[i] = (TYPE)0.0;
    xnew[i] = (TYPE)0.0;
    b[i] = (TYPE)(rand() % 51) / 100.0;
  }

  start_time = omp_get_wtime();
  //
  // jacobi iterative solver
  //
  conv = LARGE;
  iters = 0;

  while ((conv > TOLERANCE) && (iters < MAX_ITERS)) {
    iters++;

#pragma omp parallel for simd
    for (int i = 0; i < Ndim; i++) {
      xnew[i] = (TYPE)0.0;
      for (int j = 0; j < Ndim; j++) {
          xnew[i] += A[i * Ndim + j] * xold[j] * (i != j);
      }
      xnew[i] = (b[i] - xnew[i]) / A[i * Ndim + i];
    }
    //
    // test convergence
    //
    conv = 0.0;

#pragma omp parallel for simd reduction(+: conv)
    for (int i = 0; i < Ndim; i++) {
      TYPE tmp = xnew[i] - xold[i];
      conv += tmp * tmp;
    }
    conv = sqrt((double)conv);

#ifdef DEBUG
    printf(" conv = %f \n", (float)conv);
#endif

    TYPE* tmp = xold;
    xold = xnew;
    xnew = tmp;
  }

  elapsed_time = omp_get_wtime() - start_time;
  printf(" Convergence = %g with %d iterations and %f seconds\n", (float)conv,
      iters, (float)elapsed_time);
  printf("Avg seconds of iter: %f\n", elapsed_time / iters);


  //
  // test answer by multiplying my computed value of x by
  // the input A matrix and comparing the result with the
  // input b vector.
  //
  err = (TYPE)0.0;
  chksum = (TYPE)0.0;

  for (int i = 0; i < Ndim; i++) {
    xold[i] = (TYPE)0.0;
    for (int j = 0; j < Ndim; j++)
      xold[i] += A[i * Ndim + j] * xnew[j];
    TYPE tmp = xold[i] - b[i];
#ifdef DEBUG
    printf(" i=%d, diff = %f,  computed b = %f, input b= %f \n", i, (float)tmp,
        (float)xold[i], (float)b[i]);
#endif
    chksum += xnew[i];
    err += tmp * tmp;
  }
  err = sqrt((double)err);
  printf("jacobi solver: err = %f, solution checksum = %f \n", (float)err,
      (float)chksum);
  if (err > TOLERANCE)
    printf("\nWARNING: final solution error > %g\n\n", TOLERANCE);

  free(A);
  free(b);
  free(xold);
  free(xnew);
}

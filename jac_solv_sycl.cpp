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

#include "mm_utils.h" //a library of basic matrix utilities functions
#include <math.h>
// #include <omp.h>
#include <stdlib.h>
#include <chrono>
#include <sycl/sycl.hpp>
// and some key constants used in this program
//(such as TYPE)
extern SYCL_EXTERNAL int rand(void);

#define TOLERANCE 0.001
#define DEF_SIZE 1000
#define MAX_ITERS 100000
#define LARGE 1000000.0

#define DEBUG    1     // output a small subset of intermediate values
// #define VERBOSE  1

void init_diag_dom_near_identity_matrix_sycl(int Ndim, TYPE *A, sycl::id<1> idx, int size)
{

	int i, j;
	TYPE sum;

	// idx ranges from 0 to Ndim

	//
	// Create a random, diagonally dominant matrix.  For
	// a diagonally dominant matrix, the diagonal element
	// of each row is great than the sum of the other
	// elements in the row.  Then scale the matrix so the
	// result is near the identiy matrix.

	for (i = idx; i < Ndim; i += size)
	{
		sum = (TYPE)0.0;
		for (j = 0; j < Ndim; j++)
		{
			*(A + i * Ndim + j) = (rand() % 23) / (TYPE)1000.0;
			sum += *(A + i * Ndim + j);
		}
		*(A + i * Ndim + i) += sum;

		// scale the row so the final matrix is almost an identity matrix;wq
		for (j = 0; j < Ndim; j++)
			*(A + i * Ndim + j) /= sum;
	}
}

void jalc_solver_main(sycl::nd_item<1> item, int Ndim, TYPE* A, TYPE* b, TYPE* d_xnew, TYPE* d_xold, TYPE* d_conv, sycl::stream out)
{
	// since global size is the same as work group size, we can use local id as index
	int idx = item.get_local_id(0);

	*d_conv = 0.0;
	d_xnew[idx] = 0.0;

	for (int j = 0; j < Ndim; j++)
	{
		if (idx != j)
			d_xnew[idx] += A[idx * Ndim + j] * d_xold[j];
	}
	d_xnew[idx] = (b[idx] - d_xnew[idx]) / A[idx * Ndim + idx];

	item.barrier();

	// test convergence
	TYPE tmp = d_xnew[idx] - d_xold[idx];
	tmp = tmp * tmp;

	*d_conv = sycl::reduce_over_group(item.get_group(), tmp, sycl::plus<TYPE>());
}

int main(int argc, char **argv)
{
	int Ndim; // A[Ndim][Ndim]
	int iters;
	double start_time, elapsed_time;
	TYPE err, chksum;
	TYPE *A, *b, *xnew, *xold;
	TYPE *d_A, *d_b, *d_xnew, *d_xold;

	// cpu queue
	sycl::queue q;
	std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << "\n";

	// start timing
	auto start = std::chrono::steady_clock::now();

	// set matrix dimensions and allocate memory for matrices
	if (argc == 2)
	{
		Ndim = atoi(argv[1]);
	}
	else
	{
		Ndim = DEF_SIZE;
	}

	printf(" ndim = %d\n", Ndim);

	A = (TYPE *)malloc(Ndim * Ndim * sizeof(TYPE));
	b = (TYPE *)malloc(Ndim * sizeof(TYPE));
	xnew = (TYPE *)malloc(Ndim * sizeof(TYPE));
	xold = (TYPE *)malloc(Ndim * sizeof(TYPE));

	// malloc on device
	// d_A = sycl::malloc_device<TYPE>(Ndim * Ndim, q);
	// d_b = sycl::malloc_device<TYPE>(Ndim, q);
	// d_xnew = sycl::malloc_device<TYPE>(Ndim, q);
	// d_xold = sycl::malloc_device<TYPE>(Ndim, q);

	if (!A || !b || !xold || !xnew)
	{
		printf("\n memory allocation error\n");
		exit(-1);
	}

	// q.memcpy(d_A, A, Ndim * Ndim * sizeof(TYPE));

	// // generate our diagonally dominant matrix, A
	// q.submit([&](sycl::handler& h) {
	// 	// sycl::stream out(1024, 256, h); //output buffer

	// 	h.parallel_for(sycl::range<1>(8), [=](sycl::id<1> idx)
	// 	{
	// 		init_diag_dom_near_identity_matrix_sycl(Ndim, d_A, idx, 8);
	// 	});
	// });

	// q.wait();
	// q.memcpy(A, d_A, Ndim * Ndim * sizeof(TYPE)).wait();

	// std::cout << A[0] << "\n";
	init_diag_dom_near_identity_matrix(Ndim, A);
	// std::cout << A[0] << "\n";

#ifdef VERBOSE
	mm_print(Ndim, Ndim, A);
#endif

	//
	// Initialize x and just give b some non-zero random values
	//
	for (int i = 0; i < Ndim; i++)
	{
		xnew[i] = (TYPE)0.0;
    	xold[i] = (TYPE)0.0;
		b[i] = (TYPE)(rand() % 51) / 100.0;
	}

	// q.memcpy(d_A, A, Ndim * Ndim * sizeof(TYPE));
	// q.memcpy(d_b, b, Ndim * sizeof(TYPE));
	// q.memset(d_xnew, 0, Ndim * sizeof(TYPE));
	// q.memset(d_xold, 0, Ndim * sizeof(TYPE));

	// start_time = omp_get_wtime();
	//
	// jacobi iterative solver
	//
	TYPE conv = LARGE;
	// TYPE *d_conv = sycl::malloc_device<TYPE>(1, q);
	iters = 0;

	sycl::buffer<TYPE> A_buf(A, sycl::range<1>(Ndim * Ndim));
    sycl::buffer<TYPE> b_buf(b, sycl::range<1>(Ndim));
    sycl::buffer<TYPE> xold_buf(xold, sycl::range<1>(Ndim));
    sycl::buffer<TYPE> xnew_buf(xnew, sycl::range<1>(Ndim));
    sycl::buffer<TYPE> conv_buf(&conv, sycl::range<1>(1));

	while ((conv > TOLERANCE) && (iters < MAX_ITERS))
	{
		iters++;

		// q.memcpy(d_xnew, xnew, Ndim * sizeof(TYPE));
		// q.memcpy(d_xold, xold, Ndim * sizeof(TYPE));

		if ((iters & 1) == 0)
		{
			q.submit([&](sycl::handler &h)
			{
				sycl::stream out(1024, 256, h); //output buffer
				sycl::accessor A_acc(A_buf, h);
				sycl::accessor b_acc(b_buf, h);
				sycl::accessor xold_acc(xold_buf, h);
				sycl::accessor xnew_acc(xnew_buf, h);
				sycl::accessor conv_acc(conv_buf, h);

				h.parallel_for(sycl::nd_range<1>(sycl::range<1>(Ndim), sycl::range<1>(Ndim)), [=](sycl::nd_item<1> item) 
				{	
					// since global size is the same as work group size, we can use local id as index
					int idx = item.get_local_id(0);

					xnew_acc[idx] = 0.0;

					for (int j = 0; j < Ndim; j++)
					{
						if (idx != j)
							xnew_acc[idx] += A_acc[idx * Ndim + j] * xold_acc[j];
					}
					xnew_acc[idx] = (b_acc[idx] - xnew_acc[idx]) / A_acc[idx * Ndim + idx];

					item.barrier();

					// test convergence
					TYPE tmp = xnew_acc[idx] - xold_acc[idx];
					tmp = tmp * tmp;

					conv_acc[0] = sycl::reduce_over_group(item.get_group(), tmp, sycl::plus<TYPE>());
				}); 
			});
		}
		else
		{
			q.submit([&](sycl::handler &h)
			{
				sycl::stream out(1024, 256, h); //output buffer
				sycl::accessor A_acc(A_buf, h);
				sycl::accessor b_acc(b_buf, h);
				sycl::accessor xold_acc(xold_buf, h);
				sycl::accessor xnew_acc(xnew_buf, h);
				sycl::accessor conv_acc(conv_buf, h);

				h.parallel_for(sycl::nd_range<1>(sycl::range<1>(Ndim), sycl::range<1>(Ndim)), [=](sycl::nd_item<1> item) 
				{	
					// since global size is the same as work group size, we can use local id as index
					int idx = item.get_local_id(0);

					xold_acc[idx] = 0.0;

					for (int j = 0; j < Ndim; j++)
					{
						if (idx != j)
							xold_acc[idx] += A_acc[idx * Ndim + j] * xnew_acc[j];
					}
					xold_acc[idx] = (b_acc[idx] - xold_acc[idx]) / A_acc[idx * Ndim + idx];

					item.barrier();

					// test convergence
					TYPE tmp = xold_acc[idx] - xnew_acc[idx];
					tmp = tmp * tmp;

					conv_acc[0] = sycl::reduce_over_group(item.get_group(), tmp, sycl::plus<TYPE>());
				}); 
			});
		}

		q.wait();
		// q.memcpy(&conv, d_conv, 1 * sizeof(TYPE)).wait();

		conv = sqrt((double)conv);

		// for (int i = 0; i < Ndim; i++) {
		// 	xnew[i] = (TYPE)0.0;
		// 	for (int j = 0; j < Ndim; j++) {
		// 		if (i != j)
		// 		xnew[i] += A[i * Ndim + j] * xold[j];
		// 	}
		// 	xnew[i] = (b[i] - xnew[i]) / A[i * Ndim + i];
		// }
		//
		// test convergence
		//
		// conv = 0.0;
		// for (int i = 0; i < Ndim; i++) {
		// 	TYPE tmp = xnew[i] - xold[i];
		// 	conv += tmp * tmp;
		// }
		// conv = sqrt((double)conv);
#ifdef DEBUG
		printf(" conv = %f \n", (float)conv);
#endif

		// TYPE* tmp = xold;
		// xold = xnew;
		// xnew = tmp;
	}
	// elapsed_time = omp_get_wtime() - start_time;
	printf(" Convergence = %g with %d iterations and %f seconds\n", (float)conv,
		   iters, (float)elapsed_time);

	//
	// test answer by multiplying my computed value of x by
	// the input A matrix and comparing the result with the
	// input b vector.
	//
	err = (TYPE)0.0;
	chksum = (TYPE)0.0;
	// q.memcpy(xnew, d_xnew, Ndim * sizeof(TYPE));
	// q.memcpy(xold, d_xold, Ndim * sizeof(TYPE)).wait();

	for (int i = 0; i < Ndim; i++)
	{
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

	// stop timing
	auto end = std::chrono::steady_clock::now();

	printf("jacobi solver: err = %f, solution checksum = %f \n", (float)err,
		   (float)chksum);
	if (err > TOLERANCE)
		printf("\nWARNING: final solution error > %g\n\n", TOLERANCE);

	// print elapsed time
	std::chrono::duration<double> elapsed_seconds = end - start;
	printf("elapsed time: %f seconds\n", elapsed_seconds.count());

	free(A);
	free(b);
	free(xold);
	free(xnew);
	// sycl::free(d_A, q);
	// sycl::free(d_b, q);
	// sycl::free(d_xold, q);
	// sycl::free(d_xnew, q);
	// sycl::free(d_conv, q);
}

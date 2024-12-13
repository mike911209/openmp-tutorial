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

int round_up_power_of_2(int v)
{
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v++;

	return v;
}

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

void jac_solver_first(sycl::nd_item<1> item, int Ndim, int Ndim_round, TYPE* A, TYPE* d_xold, TYPE* A_reduce, int world_size)
{
	// blockIdx.x * blockDim.x + threadIdx.x
	int rank = item.get_group(0) * item.get_local_range().get(0) + item.get_local_id(0);
	TYPE test;

	// out << rank << " " << world_size << " " << Ndim << " " << Ndim_round << "\n";

	// *d_conv = 0.0;
	for (int idx = rank; idx < Ndim * Ndim; idx += world_size)
	{
		int i = idx / Ndim;
		int j = idx % Ndim;

		// d_xnew[idx] = 0.0;

		if (i != j)
			A_reduce[i * Ndim_round + j] = A[i * Ndim + j] * d_xold[j];
		else
			A_reduce[i * Ndim_round + i] = 0;
	}
}

void jac_solver_last(sycl::nd_item<1> item, int Ndim, TYPE* A, TYPE* b, TYPE* d_xnew, TYPE* d_xold, TYPE* d_conv_reduce)
{
	// blockIdx.x * blockDim.x + threadIdx.x
	int idx = item.get_group(0) * item.get_local_range().get(0) + item.get_local_id(0);
	TYPE tmp;

	if (idx < Ndim)
	{
		d_xnew[idx] = (b[idx] - d_xnew[idx]) / A[idx * Ndim + idx];

		// test convergence
		tmp = d_xnew[idx] - d_xold[idx];
		tmp = tmp * tmp;
	}
	else
	{
		tmp = 0.0;
	}

	sycl::group_barrier(item.get_group());

	d_conv_reduce[item.get_group(0)] = sycl::reduce_over_group(item.get_group(), tmp, sycl::plus<TYPE>());
}

void reduce(sycl::nd_item<1> item, TYPE *d_val, TYPE *d_arr_reduce, int size)
{
	// reduce d_arr_reduce
	int idx = item.get_local_id(0);
	int cal_size = size / 2;

	while (cal_size > 0)
	{
		if (idx < cal_size)
		{
			d_arr_reduce[idx] += d_arr_reduce[idx + cal_size];
		}
		cal_size /= 2;

		sycl::group_barrier(item.get_group());
	}

	if (idx == 0) d_val[0] = d_arr_reduce[0];
}

void reduce_A(sycl::nd_item<1> item, int i, TYPE *d_val, TYPE *d_arr_reduce, int size)
{
	// reduce d_arr_reduce
	int idx = item.get_local_id(0);
	int cal_size = size / 2;

	while (cal_size > 0)
	{
		if (idx < cal_size)
		{
			d_arr_reduce[i * size + idx] += d_arr_reduce[i * size + idx + cal_size];
		}
		cal_size /= 2;

		sycl::group_barrier(item.get_group());
	}

	if (idx == 0) d_val[i] = d_arr_reduce[i * size];
}

int main(int argc, char **argv)
{
	int Ndim, Ndim_round; // A[Ndim][Ndim]
	int iters;
	double start_time, elapsed_time;
	TYPE err, chksum;
	TYPE *A, *b, *xnew, *xold;
	TYPE *d_A, *d_A_reduce, *d_b, *d_xnew, *d_xold;

	// cpu queue
	// sycl::queue q;
	// std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << "\n";

	// gpu queue
	sycl::device d(sycl::gpu_selector_v);
	sycl::context c(d);
	sycl::queue q(c, d, sycl::property::queue::in_order{});
	sycl::queue q2(c, d);
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

	Ndim_round = round_up_power_of_2(Ndim);
	printf(" ndim = %d\n", Ndim);
	printf(" ndim Round = %d\n", Ndim_round);

	A = (TYPE *)malloc(Ndim * Ndim * sizeof(TYPE));
	b = (TYPE *)malloc(Ndim * sizeof(TYPE));
	xnew = (TYPE *)malloc(Ndim * sizeof(TYPE));
	xold = (TYPE *)malloc(Ndim * sizeof(TYPE));
	// TYPE A[Ndim * Ndim];
	// TYPE b[Ndim];
	// TYPE xnew[Ndim];
	// TYPE xold[Ndim];

	// malloc on device
	d_A = sycl::malloc_device<TYPE>(Ndim * Ndim, q);
	d_A_reduce = sycl::malloc_device<TYPE>(Ndim_round * Ndim_round, q);
	d_b = sycl::malloc_device<TYPE>(Ndim, q);
	d_xnew = sycl::malloc_device<TYPE>(Ndim, q);
	d_xold = sycl::malloc_device<TYPE>(Ndim, q);

	// if (!A || !b || !xold || !xnew)
	// {
	// 	printf("\n memory allocation error\n");
	// 	exit(-1);
	// }

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

	q.memcpy(d_A, A, Ndim * Ndim * sizeof(TYPE));
	q.memcpy(d_b, b, Ndim * sizeof(TYPE));
	q.memset(d_xnew, 0, Ndim * sizeof(TYPE));
	q.memset(d_xold, 0, Ndim * sizeof(TYPE));

	// start_time = omp_get_wtime();
	//
	// jacobi iterative solver
	//
	TYPE conv = LARGE;
	int threads_per_block = 256;
	int blocks_per_grid = ceil(Ndim / (TYPE) threads_per_block);
	int reduce_blocks_per_grid = round_up_power_of_2(blocks_per_grid);	// round up to the nearest power of 2

	std::cout << "blocks_per_grid: " << blocks_per_grid << " " << "reduce_blocks_per_grid: " << reduce_blocks_per_grid << "\n";

	TYPE *d_conv = sycl::malloc_device<TYPE>(1, q);
	TYPE *d_conv_reduce = sycl::malloc_device<TYPE>(reduce_blocks_per_grid, q);
	sycl::range<1> blocks(blocks_per_grid);
	sycl::range<1> threads(threads_per_block);

	q.memset(d_conv, LARGE, sizeof(TYPE));
	q.memset(d_conv_reduce, 0, sizeof(TYPE) * reduce_blocks_per_grid);
	q.memset(d_A_reduce, 0, Ndim_round * Ndim_round * sizeof(TYPE));
	iters = 0;

	while ((conv > TOLERANCE) && (iters < MAX_ITERS))
	{
		iters++;

		// q.memcpy(d_xnew, xnew, Ndim * sizeof(TYPE));
		// q.memcpy(d_xold, xold, Ndim * sizeof(TYPE));

		if ((iters & 1) == 0)
		{
			q.submit([&](sycl::handler &h)
			{
				// sycl::stream out(1024, 256, h); //output buffer

				h.parallel_for(sycl::nd_range<1>(blocks * threads, threads), [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]]
				{	
					// jalc_solver_main(item, Ndim, d_A, d_b, d_xnew, d_xold, d_conv, d_conv_reduce);
					jac_solver_first(item, Ndim, Ndim_round, d_A, d_xold, d_A_reduce, blocks_per_grid * threads_per_block);

					// out << d_A_reduce[0] << "\n";
				}); 
			});

			// q.wait();

			
			q.submit([&](sycl::handler &h)
			{
				// sycl::stream out(1024, 256, h); //output buffer

				h.parallel_for(sycl::nd_range<1>(Ndim_round / 2, Ndim_round / 2), [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]]
				{	
					for (int i = 0; i < Ndim; i++)
						reduce(item, d_xnew + i, d_A_reduce + i * Ndim_round, Ndim_round);
				}); 
			});
			

			// q.submit([&](sycl::handler &h)
			// {
			// 	sycl::stream out(1024, 256, h); //output buffer

			// 	h.parallel_for(sycl::nd_range<1>(Ndim, Ndim), [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]]
			// 	{	
			// 		int idx = item.get_local_id(0);
			// 		d_xnew[idx] = 0.0;

			// 		for (int i = 0; i < Ndim; i++)
			// 		{
			// 			d_xnew[idx] += d_A_reduce[idx * Ndim_round + i];
			// 			if (idx == 0) out << d_xnew[idx] << "\n";
			// 		}
			// 	}); 
			// });

			// q2.wait();	// wait for all reduction over

			q.submit([&](sycl::handler &h)
			{
				// sycl::stream out(1024, 256, h); //output buffer

				h.parallel_for(sycl::nd_range<1>(blocks * threads, threads), [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]]
				{	
					jac_solver_last(item, Ndim, d_A, d_b, d_xnew, d_xold, d_conv_reduce);
				}); 
			});

			q.submit([&](sycl::handler &h)
			{
				// sycl::stream out(1024, 256, h); //output buffer

				h.parallel_for(sycl::nd_range<1>(reduce_blocks_per_grid / 2, reduce_blocks_per_grid / 2), [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]]
				{	
					reduce(item, d_conv, d_conv_reduce, reduce_blocks_per_grid);
				}); 
			});
		}
		else
		{
			q.submit([&](sycl::handler &h)
			{
				// sycl::stream out(1024, 256, h); //output buffer

				h.parallel_for(sycl::nd_range<1>(blocks * threads, threads), [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]]
				{	
					// jalc_solver_main(item, Ndim, d_A, d_b, d_xold, d_xnew, d_conv, d_conv_reduce);
					jac_solver_first(item, Ndim, Ndim_round, d_A, d_xnew, d_A_reduce, blocks_per_grid * threads_per_block);
				}); 
			});

			// q.wait();

			q.submit([&](sycl::handler &h)
			{
				// sycl::stream out(1024, 256, h); //output buffer

				h.parallel_for(sycl::nd_range<1>(Ndim_round / 2, Ndim_round / 2), [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]]
				{	
					for (int i = 0; i < Ndim; i++)
						reduce(item, d_xold + i, d_A_reduce + i * Ndim_round, Ndim_round);
				}); 
			});

			// q.submit([&](sycl::handler &h)
			// {
			// 	h.parallel_for(sycl::nd_range<1>(Ndim, Ndim), [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]]
			// 	{	
			// 		int idx = item.get_local_id(0);
			// 		d_xold[idx] = 0.0;

			// 		for (int i = 0; i < Ndim; i++)
			// 		{
			// 			d_xold[idx] += d_A_reduce[idx * Ndim_round + i];
			// 		}
			// 	}); 
			// });

			// q2.wait();	// wait for all reduction over

			q.submit([&](sycl::handler &h)
			{
				// sycl::stream out(1024, 256, h); //output buffer

				h.parallel_for(sycl::nd_range<1>(blocks * threads, threads), [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]]
				{	
					jac_solver_last(item, Ndim, d_A, d_b, d_xold, d_xnew, d_conv_reduce);
				}); 
			});

			q.submit([&](sycl::handler &h)
			{
				// sycl::stream out(1024, 256, h); //output buffer

				h.parallel_for(sycl::nd_range<1>(reduce_blocks_per_grid / 2, reduce_blocks_per_grid / 2), [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]]
				{	
					reduce(item, d_conv, d_conv_reduce, reduce_blocks_per_grid);
				}); 
			});
		}

		q.wait();
		q.memcpy(&conv, d_conv, 1 * sizeof(TYPE)).wait();

		conv = sqrt((double)conv);

		//
		// test convergence
		//
		// conv = 0.0;
		// if ((iters & 1) == 0)
		// {
		// 	for (int i = 0; i < Ndim; i++) {
		// 		TYPE tmp = xnew[i] - xold[i];
		// 		conv += tmp * tmp;
		// 	}
		// }
		// else
		// {
		// 	for (int i = 0; i < Ndim; i++) {
		// 		TYPE tmp = xold[i] - xnew[i];
		// 		conv += tmp * tmp;
		// 	}
		// }

		// conv = sqrt((double)conv);
#ifdef DEBUG
		printf(" conv = %f \n", (float)conv);
#endif

		// TYPE* tmp = xold;
		// xold = xnew;
		// xnew = tmp;

		// if (iters == 2)	break;
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
	q.memcpy(xnew, d_xnew, Ndim * sizeof(TYPE));
	q.memcpy(xold, d_xold, Ndim * sizeof(TYPE)).wait();

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

	// free(A);
	// free(b);
	// free(xold);
	// free(xnew);
	// sycl::free(d_A, q);
	// sycl::free(d_b, q);
	// sycl::free(d_xold, q);
	// sycl::free(d_xnew, q);
	// sycl::free(d_conv, q);
}

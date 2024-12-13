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

#define _STR(X) #X
#define STR(X) _STR(X)

#define _TYPE double

using TYPE = _TYPE;

void init_diag_dom_near_identity_matrix(const int Ndim, TYPE *const A)
{
    // Create a random, diagonally dominant matrix.  For
    // a diagonally dominant matrix, the diagonal element
    // of each row is great than the sum of the other
    // elements in the row.  Then scale the matrix so the
    // result is near the identiy matrix.
    static std::mt19937 rng(std::random_device{}());
    static std::uniform_int_distribution<int> uid(0, RAND_MAX);
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

#include <CL/cl.hpp>

constexpr double TOLERANCE = 0.001,
                 LARGE = 1000000.0;
constexpr int DEF_SIZE = 1024,
              MAX_ITERS = 65536;

// #define DEBUG    1     // output a small subset of intermediate values
// #define VERBOSE  1

std::vector<cl::Device> get_device_list()
{
    std::vector<cl::Platform> plats;
    cl::Platform::get(&plats);
    cl::Platform &plat = plats.front();
    std::vector<cl::Device> devs;
    plat.getDevices(CL_DEVICE_TYPE_GPU, &devs);
    return devs;
}

std::string get_kernel_string()
{
    std::stringstream ss;
    std::ifstream fin("jac_solv.cl");
    std::string s;
    while (std::getline(fin, s))
        ss << s << '\n';
    return ss.str();
}

int main(int argc, char **argv)
{
    // set matrix dimensions and allocate memory for matrices
    const int Ndim = argc > 1 ? atoi(argv[1]) : DEF_SIZE, // A[Ndim][Ndim]
              wgsize = argc > 2 ? atoi(argv[2]) : 0,
              conv_wgsize = argc > 2 ? atoi(argv[2]) : 64,
              dev_idx = argc > 3 ? atoi(argv[3]) : 0;
    TYPE err, chksum,
         *const A = new TYPE[Ndim * Ndim],
         *const b = new TYPE[Ndim],
         *xnew = new TYPE[Ndim],
         *xold = new TYPE[Ndim],
         *const conv_temp = new TYPE[Ndim / conv_wgsize];

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
        b[i] = (rand() % 51) / 100.0;
    }

    const cl::Device dev = get_device_list().front();
    std::cout << "\nUsing OpenCL device: " << dev.getInfo<CL_DEVICE_NAME>() << '\n';
    
    const cl::Context ctx({dev});
    const cl::CommandQueue cmdQ(ctx, dev);

    const cl::Program prog(ctx, get_kernel_string());
    prog.build("-DTYPE="STR(_TYPE));

    cl::Kernel jacobi(prog, "jacobi"),
               convergence(prog, "convergence");
    
    const cl::Buffer d_A(ctx, CL_MEM_READ_ONLY, sizeof(TYPE) * Ndim * Ndim),
                     d_b(ctx, CL_MEM_READ_ONLY, sizeof(TYPE) * Ndim),
                     d_conv(ctx, CL_MEM_WRITE_ONLY, sizeof(TYPE) * (Ndim / conv_wgsize));
     cl::Buffer d_xnew(ctx, CL_MEM_READ_WRITE, sizeof(TYPE) * Ndim),
                d_xold(ctx, CL_MEM_READ_WRITE, sizeof(TYPE) * Ndim);
    
    cmdQ.enqueueWriteBuffer(d_A, CL_FALSE, 0, sizeof(TYPE) * Ndim * Ndim, A);
    cmdQ.enqueueWriteBuffer(d_b, CL_FALSE, 0, sizeof(TYPE) * Ndim, b);
    cmdQ.enqueueWriteBuffer(d_xnew, CL_FALSE, 0, sizeof(TYPE) * Ndim, xnew);
    cmdQ.enqueueWriteBuffer(d_xold, CL_FALSE, 0, sizeof(TYPE) * Ndim, xold);

    jacobi.setArg(0, Ndim);
    jacobi.setArg(1, d_A);
    jacobi.setArg(2, d_b);

    convergence.setArg(0, d_xnew);
    convergence.setArg(1, d_xold);
    convergence.setArg(2, d_conv);
    convergence.setArg(3, sizeof(TYPE) * conv_wgsize, nullptr);

    const auto start_time = std::chrono::steady_clock::now();
    //
    // jacobi iterative solver
    //
    TYPE conv = LARGE;
    int iters = 0;
    while (conv > TOLERANCE && iters < MAX_ITERS)
    {
        iters++;

        jacobi.setArg(3, d_xold);
        jacobi.setArg(4, d_xnew);

        cmdQ.enqueueNDRangeKernel(jacobi, cl::NullRange, cl::NDRange(Ndim), wgsize ? cl::NDRange(wgsize) : cl::NullRange);
        //
        // test convergence
        //
        cmdQ.enqueueNDRangeKernel(convergence, cl::NullRange, cl::NDRange(Ndim), cl::NDRange(conv_wgsize));
        cmdQ.enqueueReadBuffer(d_conv, CL_TRUE, 0, sizeof(TYPE) * (Ndim / conv_wgsize), conv_temp);
        conv = sqrt(std::accumulate(conv_temp, conv_temp + Ndim / conv_wgsize, 0.));

        std::swap(d_xnew, d_xold);
    }
    cmdQ.finish();
    const auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time);
    std::cout << " Convergence = " << conv << " with " << iters << " iterations and " << elapsed_time.count() << " seconds\n";

    //
    // test answer by multiplying my computed value of x by
    // the input A matrix and comparing the result with the
    // input b vector.
    //
    cmdQ.enqueueReadBuffer(d_xnew, CL_FALSE, 0, sizeof(TYPE) * Ndim, xnew);
    cmdQ.enqueueReadBuffer(d_xold, CL_TRUE, 0, sizeof(TYPE) * Ndim, xold);
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

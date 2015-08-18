#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <clFFT.h>
#include <fftw3.h>


static const char *kernelSource =
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable  \n" \
"__kernel void mult(__global double *v) {       \n" \
"    int id, v_re, v_im;       \n" \
"    id   = get_global_id(0);  \n" \
"    v_re = 2*id;              \n" \
"    v_im = v_re + 1;          \n" \
"                              \n" \
"    v[v_re] = 2*v[v_re];      \n" \
"    v[v_im] = 4*v[v_im];      \n" \
"}                             \n" \
"\n" ;


int roundUpToNearest(int x, int n) {
    /* Rounds x UP to nearest multiple of n. */
    int x_rem = x % n;
    if (x_rem == 0)
        return x;

    return x + (n - x_rem);
}


void checkIfArraysEqual(double *h_v, double *v, int N, double epsilon) {
    int arrays_equal = 1;
    int i;

    for (i=0; i<N; i++) {
        // printf("[%f %f]  ", h_v[i], v[i]);
        if (abs(v[i] - h_v[i]) > epsilon)
            arrays_equal = 0;
    }
    
    if (arrays_equal)
        printf("Arrays are equal!\n");
    else
        printf("Arrays are NOT equal!\n");
}

 
int main( int argc, char* argv[] ) {
    /* This setup is a bit tricky. Since we're doing a real transform, CLFFT
     * requires N+2 elements in the array. This is because only N/2 + 1 numbers
     * are calculated, and since each number is complex, it requires 2 elements
     * for space.
     *
     * To avoid warp divergence, we want to avoid any conditionals in the
     * kernel. Thus we cannot check to see if the thread ID is even or odd to
     * act on a real number or imaginary number. To do this, one thread should
     * handle one complex number (one real, one imag), i.e. ID_j should handle
     * array elements j, j+1.
     *
     * But we also need the number of global items to be a multiple of 32 (warp
     * size). What we can do, for example, N = 128, is pad it by 2 (130),
     * divide it by 2 (65), round that UP to the nearest 32 (96), multiply that
     * by 2 (192). The kernel will operate on zeros, but it should be faster
     * than the scenario with warp divergence. */

    unsigned int N = 4096;
    unsigned int N_pad = 2*roundUpToNearest( (N+2)/2, 32 );
    size_t N_bytes = N_pad * sizeof(double);

    // openCL declarations
    cl_platform_id platform;
    cl_device_id device_id;
    cl_context context; 
    cl_command_queue queue;
    cl_program program;
    cl_kernel k_mult;

    // clFFT declarations
    clfftPlanHandle planHandleForward, planHandleBackward;
    clfftDim dim = CLFFT_1D;
    size_t clLengths[1] = {N};
    clfftSetupData fftSetup;
    clfftInitSetupData(&fftSetup);
    clfftSetup(&fftSetup);
 
    // host version of v
    double *h_v;
    h_v = (double*) malloc(N_bytes);
 
    // initialize v on host (GPU and CPU)
    int i;
    for (i = 0; i < N; i++)
        h_v[i] = i;

    // CPU TRANSFORM ----------------------------------------------------------
    double *v;
    fftw_complex *V;
    int N_COMPLEX = N/2 + 1;
    int REAL = 0;
    int IMAG = 1;

    v = (double*) malloc(N * sizeof(double));
    V = (fftw_complex*) malloc(N_COMPLEX * sizeof(fftw_complex));

    fftw_plan fft  = fftw_plan_dft_r2c_1d(N, v, V, FFTW_MEASURE);
    fftw_plan ifft = fftw_plan_dft_c2r_1d(N, V, v, FFTW_MEASURE);

    // initialize v here because otherwise fftw_execute will run before we 
    // initialize the plan... for some reason.
    for (i=0; i<N; i++)
        v[i] = i;

    fftw_execute(fft);
    for (i=0; i<N_COMPLEX; i++) {
        V[i][REAL] = 2 * V[i][REAL];
        V[i][IMAG] = 4 * V[i][IMAG];
    }
    fftw_execute(ifft);

    // scale array as FFTW doesn't automatically do this for back transform
    for (i=0; i<N; i++)
        v[i] = v[i]/N;


    // GPU STUFF --------------------------------------------------------------
    // global & local number of threads
    size_t globalSize, localSize;
    globalSize = N_pad / 2;
    localSize = 32;

    // setup OpenCL stuff 
    cl_int err;
    err = clGetPlatformIDs(1, &platform, NULL);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device_id, 0, &err);
    program = clCreateProgramWithSource(context, 1, (const char **) & kernelSource, NULL, &err);
 
    // Build the program executable 
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("building program failed\n");
        if (err == CL_BUILD_PROGRAM_FAILURE) {
            size_t log_size;
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
            char *log = (char *) malloc(log_size);
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            printf("%s\n", log);
        }
    }
    k_mult = clCreateKernel(program, "mult", &err);
 
    // create arrays on host and write them
    cl_mem d_v, d_V;
    d_v = clCreateBuffer(context, CL_MEM_READ_WRITE, N_bytes, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, d_v, CL_TRUE, 0, N_bytes, h_v, 0, NULL, NULL);

    // REAL IN-PLACE TRANSFORM ------------------------------------------------
    // create forward plan and set its params
    clfftCreateDefaultPlan(&planHandleForward, context, dim, clLengths);
    clfftSetPlanPrecision(planHandleForward, CLFFT_DOUBLE);
    clfftSetLayout(planHandleForward, CLFFT_REAL, CLFFT_HERMITIAN_INTERLEAVED);
    clfftSetResultLocation(planHandleForward, CLFFT_INPLACE);
    clfftBakePlan(planHandleForward, 1, &queue, NULL, NULL);

    // create backward plan and set its params
    clfftCreateDefaultPlan(&planHandleBackward, context, dim, clLengths);
    clfftSetPlanPrecision(planHandleBackward, CLFFT_DOUBLE);
    clfftSetLayout(planHandleBackward, CLFFT_HERMITIAN_INTERLEAVED, CLFFT_REAL);
    clfftSetResultLocation(planHandleBackward, CLFFT_INPLACE);
    clfftBakePlan(planHandleBackward, 1, &queue, NULL, NULL);

    err  = clSetKernelArg(k_mult, 0, sizeof(cl_mem), &d_v);
 
    // FFT data, multiply elements, IFFT data
    clfftEnqueueTransform(planHandleForward, CLFFT_FORWARD, 1, &queue, 0, NULL, NULL, &d_v, NULL, NULL);
    clFinish(queue);
    err = clEnqueueNDRangeKernel(queue, k_mult, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
    clFinish(queue);
    clfftEnqueueTransform(planHandleBackward, CLFFT_BACKWARD, 1, &queue, 0, NULL, NULL, &d_v, NULL, NULL);
    clFinish(queue);
    clEnqueueReadBuffer(queue, d_v, CL_TRUE, 0, N_bytes, h_v, 0, NULL, NULL );
    clFinish(queue);
    
    clfftDestroyPlan( &planHandleForward );
    clfftDestroyPlan( &planHandleBackward );
    printf("Testing in-place real transform... ");
    checkIfArraysEqual(h_v, v, N, 0.0);


    
    // REAL OUT-OF-PLACE TRANSFORM --------------------------------------------
    // reset array
    d_v = clCreateBuffer(context, CL_MEM_READ_WRITE, N_bytes, NULL, NULL);
    d_V = clCreateBuffer(context, CL_MEM_READ_WRITE, N_bytes, NULL, NULL);
    
    cl_mem inputBuffers[1] = {0}, outputBuffers[1] = {0};
    inputBuffers[0] = d_v;
    outputBuffers[0] = d_V;

    err = clEnqueueWriteBuffer(queue, d_v, CL_TRUE, 0, N_bytes, h_v, 0, NULL, NULL);

    clfftCreateDefaultPlan(&planHandleForward, context, dim, clLengths);
    clfftSetPlanPrecision(planHandleForward, CLFFT_DOUBLE);
    clfftSetLayout(planHandleForward, CLFFT_REAL, CLFFT_HERMITIAN_INTERLEAVED);
    clfftSetResultLocation(planHandleForward, CLFFT_OUTOFPLACE);
    clfftBakePlan(planHandleForward, 1, &queue, NULL, NULL);

    clfftCreateDefaultPlan(&planHandleBackward, context, dim, clLengths);
    clfftSetPlanPrecision(planHandleBackward, CLFFT_DOUBLE);
    clfftSetLayout(planHandleBackward, CLFFT_HERMITIAN_INTERLEAVED, CLFFT_REAL);
    clfftSetResultLocation(planHandleBackward, CLFFT_OUTOFPLACE);
    clfftBakePlan(planHandleBackward, 1, &queue, NULL, NULL);

    clfftEnqueueTransform(planHandleForward, CLFFT_FORWARD, 1, &queue, 0, NULL, NULL, &inputBuffers, &outputBuffers, NULL);
    clFinish(queue);
    err = clEnqueueNDRangeKernel(queue, k_mult, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
    clFinish(queue);
    clfftEnqueueTransform(planHandleBackward, CLFFT_BACKWARD, 1, &queue, 0, NULL, NULL, &inputBuffers, &outputBuffers, NULL);
    clFinish(queue);
    clEnqueueReadBuffer(queue, d_v, CL_TRUE, 0, N_bytes, h_v, 0, NULL, NULL );
    clFinish(queue);

    printf("Testing out-of-place transform... ");
    checkIfArraysEqual(h_v, v, N, 0.0);


    // release FFT stuff
    fftw_free(V);
    clfftDestroyPlan( &planHandleForward );
    clfftDestroyPlan( &planHandleBackward );
    clfftTeardown();
 
    // release OpenCL resources
    clReleaseMemObject(d_v);
    clReleaseProgram(program);
    clReleaseKernel(k_mult);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
 
    //release host memory
    free(v);
    free(h_v);
 
    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <clFFT.h>
 
const char *kernelSource =
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable                 \n" \ 
"__kernel void mult(__global double *vR, __global double *vI) {  \n" \
"    int id;                   \n" \
"    id = get_global_id(0);    \n" \
"    vR[id] = 2*vR[id];        \n" \
"    vI[id] = 2*vI[id];        \n" \
"}                             \n" \
"\n" ;
 
int main( int argc, char* argv[] ) {
    // problem-related declarations
    unsigned int N = 128;
    size_t N_bytes = N * sizeof(double);

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
    double *h_vR, *h_vI;  // real & imaginary parts
    h_vR = (double*) malloc(N_bytes);
    h_vI = (double*) malloc(N_bytes);
 
    // initialize v on host
    int i;
    for (i = 0; i < N; i++) {
        h_vR[i] = i;
        h_vI[i] = 2*i;
    }

    // global & local number of threads
    size_t globalSize, localSize;
    globalSize = N;
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
    cl_mem d_vR, d_vI;
    d_vR = clCreateBuffer(context, CL_MEM_READ_WRITE, N_bytes, NULL, NULL);
    d_vI = clCreateBuffer(context, CL_MEM_READ_WRITE, N_bytes, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, d_vR, CL_TRUE, 0, N_bytes, h_vR, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, d_vI, CL_TRUE, 0, N_bytes, h_vI, 0, NULL, NULL);

    // create forward plan and set its params
    clfftCreateDefaultPlan(&planHandleForward, context, dim, clLengths);
    clfftSetPlanPrecision(planHandleForward, CLFFT_DOUBLE);
    clfftSetLayout(planHandleForward, CLFFT_COMPLEX_PLANAR, CLFFT_COMPLEX_PLANAR);
    clfftSetResultLocation(planHandleForward, CLFFT_INPLACE);
    clfftBakePlan(planHandleForward, 1, &queue, NULL, NULL);

    // create backward plan and set its params
    clfftCreateDefaultPlan(&planHandleBackward, context, dim, clLengths);
    clfftSetPlanPrecision(planHandleBackward, CLFFT_DOUBLE);
    clfftSetLayout(planHandleBackward, CLFFT_COMPLEX_PLANAR, CLFFT_COMPLEX_PLANAR);
    clfftSetResultLocation(planHandleBackward, CLFFT_INPLACE);
    clfftBakePlan(planHandleBackward, 1, &queue, NULL, NULL);

    // set all of ze kernel args...
    err  = clSetKernelArg(k_mult, 0, sizeof(cl_mem), &d_vR);
    err |= clSetKernelArg(k_mult, 1, sizeof(cl_mem), &d_vI);
 
    // cl_mem array allows for complex_planar transform
    cl_mem inputBuffers[2] = {0, 0};
    inputBuffers[0] = d_vR;
    inputBuffers[1] = d_vI;
    
    // FFT data, apply psi, IFFT data
    clfftEnqueueTransform(planHandleForward, CLFFT_FORWARD, 1, &queue, 0, NULL, NULL, &inputBuffers, NULL, NULL);
    clFinish(queue);
 
     err = clEnqueueNDRangeKernel(queue, k_mult, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
     if (err != CL_SUCCESS)
         printf("oh wtf\n");

    clfftEnqueueTransform(planHandleBackward, CLFFT_BACKWARD, 1, &queue, 0, NULL, NULL, &inputBuffers, NULL, NULL);

    // transfer back
    clEnqueueReadBuffer(queue, d_vR, CL_TRUE, 0, N_bytes, h_vR, 0, NULL, NULL );
    clEnqueueReadBuffer(queue, d_vI, CL_TRUE, 0, N_bytes, h_vI, 0, NULL, NULL );
    clFinish(queue);
 
    printf("[  ");
    for (i=0; i<N; i++)
        printf("(%f, %f)  ", h_vR[i], h_vI[i]);
    printf("]\n");

    // release clFFT stuff
    clfftDestroyPlan( &planHandleForward );
    clfftDestroyPlan( &planHandleBackward );
    clfftTeardown();
 
    // release OpenCL resources
    clReleaseMemObject(d_vR);
    clReleaseMemObject(d_vI);
    clReleaseProgram(program);
    clReleaseKernel(k_mult);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
 
    //release host memory
    free(h_vR);
    free(h_vI);
 
    return 0;
}

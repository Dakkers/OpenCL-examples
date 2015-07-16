#include <stdlib.h>
#include <stdio.h>
#include <CL/cl.h>

const char *kernel_code =
    "__kernel void multiply_by(__global int* A, const int c) {"
    "   A[get_global_id(0)] = c * A[get_global_id(0)];"
    "}";


int factorial(int n) {
    return (n <= 1) ? 1 : n * factorial(n-1);
}


int main( void ) {
    // OpenCL related declarations
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
    cl_context ctx;
    cl_program program;
    cl_command_queue queue;
    cl_event event = NULL;
    cl_kernel k_multiplyby;
    int i;

    // 
    const size_t N = 1024; // vector size
    const int c_max = 5;   // max value to iterate to
    const int coeff = factorial(c_max);
    
    int *A, *B, *C;        // A is initial, B is result, C is expected result
    A = (int*) malloc(N * sizeof(*A));
    B = (int*) malloc(N * sizeof(*B));
    C = (int*) malloc(N * sizeof(*C));
    for (i=0; i<N; i++) {
        A[i] = i;
        C[i] = coeff*i;
    }
    cl_mem d_A;  // buffer object for A

    /* Setup OpenCL environment. */
    err = clGetPlatformIDs( 1, &platform, NULL );
    err = clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL );

    props[1] = (cl_context_properties)platform;
    ctx = clCreateContext( props, 1, &device, NULL, NULL, &err );
    queue = clCreateCommandQueue( ctx, device, 0, &err );
    program = clCreateProgramWithSource(ctx, 1, (const char **) &kernel_code, NULL, &err);
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    k_multiplyby = clCreateKernel(program, "multiply_by", &err);

    // initialize buffer with data
    d_A = clCreateBuffer( ctx, CL_MEM_READ_WRITE, N*sizeof(*A), NULL, &err );

    err = clEnqueueWriteBuffer( queue, d_A, CL_TRUE, 0, N*sizeof(*A), A, 0, NULL, NULL );
    
    clSetKernelArg(k_multiplyby, 0, sizeof(cl_mem), &d_A);
    int c;
    for (c=2; c<=c_max; c++) {
        clSetKernelArg(k_multiplyby, 1, sizeof(int), &c);
        clEnqueueNDRangeKernel(queue, k_multiplyby, 1, NULL, &N, &N, 0, NULL, NULL);
    }
    err = clFinish(queue);

    err = clEnqueueReadBuffer( queue, d_A, CL_TRUE, 0, N*sizeof(*B), B, 0, NULL, NULL );
    err = clFinish(queue);

    int success = 1;
    for (i=0; i<N; i++) {
        if (B[i] != C[i]) {
            success = 0;
            break;
        }
    }

    if (success)
        printf("Arrays are equal!\n");
    else
        printf("Arrays are NOT equal\n");


    /* Release OpenCL memory objects. */
    clReleaseMemObject( d_A );
    free(A);
    free(B);
    free(C);
    clReleaseCommandQueue( queue );
    clReleaseContext( ctx );

    return 0;
}

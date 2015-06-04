#include <iostream>
#include "CL/cl.hpp"
#include <ctime>


double time_add_vectors(int n, int k) {
    // adds two vectors of size n, k times, returns total duration
    std::clock_t start;
    double duration;

    int A[n], B[n], C[n];
    for (int i=0; i<n; i++) {
        A[i] = i;
        B[i] = n-i;
        C[i] = 0;
    }

    start = std::clock();
    for (int i=0; i<k; i++) {
        for (int j=0; j<n; j++) {
            C[j] = A[j] + B[j];
        }
    }

    duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;
    return duration;
}


int main() {
    // get all platforms (drivers), e.g. NVIDIA
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);

    if (all_platforms.size()==0) {
        std::cout<<" No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Platform default_platform=all_platforms[0];
    // std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";

    // get default device (CPUs, GPUs) of the default platform
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if(all_devices.size()==0){
        std::cout<<" No devices found. Check OpenCL installation!\n";
        exit(1);
    }

    // use device[1] because that's a GPU; device[0] is the CPU
    cl::Device default_device=all_devices[1];
    // std::cout<< "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";

    // a context is like a "runtime link" to the device and platform;
    // i.e. communication is possible
    cl::Context context({default_device});

    // create the program that we want to execute on the device
    cl::Program::Sources sources;

    /*
    // calculates for each element; C = A + B
    std::string kernel_code=
        "   void kernel simple_add(global const int* A, global const int* B, global int* C) {"
        "       C[get_global_id(0)] = A[get_global_id(0)] + B[get_global_id(0)];"
        "   }";
    sources.push_back({kernel_code.c_str(), kernel_code.length()});

    cl::Program program(context, sources);
    if (program.build({default_device}) != CL_SUCCESS) {
        std::cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << std::endl;
        exit(1);
    }

    // create buffers on device (allocate space on GPU)
    cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(int) * 10);
    cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(int) * 10);
    cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, sizeof(int) * 10);

    // create things on here (CPU)
    int A[] = {0,1,2,3,4,5,6,7,8,9};
    int B[] = {0,1,2,0,1,2,0,1,2,0};

    // create a queue (a queue of commands that the GPU will execute)
    cl::CommandQueue queue(context, default_device);

    // push write commands to queue
    queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(int)*10, A);
    queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(int)*10, B);

    // RUN ZE KERNEL
    cl::KernelFunctor simple_add(cl::Kernel(program, "simple_add"), queue, cl::NullRange, cl::NDRange(10), cl::NullRange);
    simple_add(buffer_A, buffer_B, buffer_C);

    int C[10];
    // read result from GPU to here
    queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(int)*10, C);

    std::cout << "result: {";
    for (int i=0; i<10; i++) {
        std::cout << C[i] << " ";
    }
    std::cout << "}" << std::endl;
    */
    return 0;
}


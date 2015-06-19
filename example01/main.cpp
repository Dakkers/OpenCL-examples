#include <iostream>
#include <ctime>
#include "../CL/cl.hpp"


void compareResults (double CPUtime, double GPUtime, int trial) {
    double time_ratio = (CPUtime / GPUtime);
    std::cout << "VERSION "   << trial   << " -----------" << std::endl;
    std::cout << "CPU time: " << CPUtime << std::endl;
    std::cout << "GPU time: " << GPUtime << std::endl;
    std::cout << "GPU is ";
    if (time_ratio > 1)
        std::cout << time_ratio << " times faster!" << std::endl;
    else
        std::cout << (1/time_ratio) << " times slower :(" << std::endl;
}


double timeAddVectorsCPU(int n, int k) {
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
        for (int j=0; j<n; j++)
            C[j] = A[j] + B[j];
    }

    duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;
    return duration;
}


int main(int argc, char* argv[]) {

    bool verbose;
    if (argc == 1 || std::strcmp(argv[1], "0") == 0)
        verbose = true;
    else
        verbose = false;
    
    const int n = 131072;                    // size of vectors (32 * 512 * 8)
    const int k = 1000;                      // number of loop iterations
    const int NUM_GLOBAL_WITEMS = 32 * 512;  // number of threads for versions 1, 2
    int constants[2] = {n, k};

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

    cl::Context context({default_device});
    cl::Program::Sources sources;

    // calculates for each element; C = A + B
    std::string kernel_code=
        //  is equivalent to the host's "time_add_vectors" function, except the
        //  timing will be done on the host.
        "   void kernel add_looped(global const int* v1, global const int* v2, global int* v3, "
        "                          global const int* constants) {"
        "       int ID, NUM_GLOBAL_WITEMS, n, k, ratio, start, stop;"
        "       ID = get_global_id(0);"
        "       NUM_GLOBAL_WITEMS = get_global_size(0);"
        "       n = constants[0];"  // size of vectors
        "       k = constants[1];"  // number of loop iterations
        ""
        "       ratio = (n / NUM_GLOBAL_WITEMS);" // elements per thread
        "       start = ratio * ID;"
        "       stop  = ratio * (ID+1);"
        ""
        "       int i, j;" // will the compiler optimize this anyway? probably.
        "       for (i=0; i<k; i++) {"
        "           for (j=start; j<stop; j++)"
        "               v3[j] = v1[j] + v2[j];"
        "       }"
        "   }"
        ""
        "   void kernel add(global const int* v1, global const int* v2, global int* v3, "
        "                   global const int* constants) {"
        "       int ID, NUM_GLOBAL_WITEMS, n, ratio, start, stop;"
        "       ID = get_global_id(0);"
        "       NUM_GLOBAL_WITEMS = get_global_size(0);"
        "       n = constants[0];"
        ""
        "       ratio = (n / NUM_GLOBAL_WITEMS);"
        "       start = ratio * ID;"
        "       stop  = ratio * (ID+1);"
        ""
        "       for (int i=start; i<stop; i++)"
        "           v3[i] = v1[i] + v2[i];"
        "   }"
        ""
        "   void kernel add_single(global const int* v1, global const int* v2, global int* v3, "
        "                          global const int* constants) { "
        "       int k  = constants[1];"
        "       int ID = get_global_id(0);"
        "       for (int i=0; i<k; i++)"
        "           v3[ID] = v1[ID] + v2[ID];"
        "   }"
        ""  // same as add_single, but with the overhead (indexing, determining ratio) of versions 01 and 02
        "   void kernel add_single_overhead(global const int* v1, global const int* v2, global int* v3,"
        "                                   global const int* constants) {"
        "       int ID, NUM_GLOBAL_WITEMS, n, k, ratio, start, stop;"
        "       ID = get_global_id(0);"
        "       NUM_GLOBAL_WITEMS = get_global_size(0);"
        "       n = constants[0];"
        "       k = constants[1];"
        ""
        "       ratio = (n / NUM_GLOBAL_WITEMS);"
        "       start = ratio * ID;"
        "       stop  = ratio * (ID+1);"
        ""
        "       for (int i=0; i<k; i++)"
        "           v3[ID] = v1[ID] + v2[ID];"
        "   }";
    sources.push_back({kernel_code.c_str(), kernel_code.length()});

    cl::Program program(context, sources);
    if (program.build({default_device}) != CL_SUCCESS) {
        std::cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << std::endl;
        exit(1);
    }

    // run the CPU code
    float CPUtime = timeAddVectorsCPU(n, k);

    // set up kernels and vectors for GPU code
    cl::CommandQueue queue(context, default_device);
    cl::Kernel add_looped = cl::Kernel(program, "add_looped");
    cl::Kernel add        = cl::Kernel(program, "add");
    cl::Kernel add_single = cl::Kernel(program, "add_single");
    cl::Kernel add_single_overhead = cl::Kernel(program, "add_single_overhead");

    // construct vectors
    int A[n], B[n], C[n];
    for (int i=0; i<n; i++) {
        A[i] = i;
        B[i] = n - i - 1;
        C[i] = 0;
    }
    std::clock_t start_time;


    // VERSION 1 ==========================================
    // start timer
    double GPUtime1;
    start_time = std::clock();

    // allocate space
    cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(int) * n);
    cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(int) * n);
    cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, sizeof(int) * n);
    cl::Buffer buffer_constants(context, CL_MEM_READ_ONLY, sizeof(int) * 2);

    // push write commands to queue
    queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(int)*n, A);
    queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(int)*n, B);
    queue.enqueueWriteBuffer(buffer_constants, CL_TRUE, 0, sizeof(int)*2, constants);

    // RUN ZE KERNEL
    add_looped.setArg(0, buffer_A);
    add_looped.setArg(1, buffer_B);
    add_looped.setArg(2, buffer_C);
    add_looped.setArg(3, buffer_constants);
    queue.enqueueNDRangeKernel(add_looped, cl::NullRange,      // kernel, offset
                               cl::NDRange(NUM_GLOBAL_WITEMS), // global number of work items
                               cl::NDRange(32));               // local number (per group)

    // read result from GPU to here; including for the sake of timing
    queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(int)*n, C);
    queue.enqueueBarrier();
    GPUtime1 = (std::clock() - start_time) / (double) CLOCKS_PER_SEC;

    // VERSION 2 ==========================================
    double GPUtime2;
    start_time = std::clock();

    cl::Buffer buffer_A2(context, CL_MEM_READ_WRITE, sizeof(int)*n);
    cl::Buffer buffer_B2(context, CL_MEM_READ_WRITE, sizeof(int)*n);
    cl::Buffer buffer_C2(context, CL_MEM_READ_WRITE, sizeof(int)*n);
    cl::Buffer buffer_constants2(context, CL_MEM_READ_ONLY, sizeof(int)*2);
    for (int i=0; i<k; i++) {
        queue.enqueueWriteBuffer(buffer_A2, CL_TRUE, 0, sizeof(int)*n, A);
        queue.enqueueWriteBuffer(buffer_B2, CL_TRUE, 0, sizeof(int)*n, B);
        queue.enqueueWriteBuffer(buffer_constants2, CL_TRUE, 0, sizeof(int)*2, constants);

        add_looped.setArg(0, buffer_A2);
        add_looped.setArg(1, buffer_B2);
        add_looped.setArg(2, buffer_C2);
        add_looped.setArg(3, buffer_constants);
        queue.enqueueNDRangeKernel(add, cl::NullRange, cl::NDRange(NUM_GLOBAL_WITEMS), cl::NDRange(32));
    }
    queue.enqueueReadBuffer(buffer_C2, CL_TRUE, 0, sizeof(int)*n, C);
    queue.enqueueBarrier();
    GPUtime2 = (std::clock() - start_time) / (double) CLOCKS_PER_SEC;

    // VERSION 3 ==========================================
    double GPUtime3;
    start_time = std::clock();

    cl::Buffer buffer_A3(context, CL_MEM_READ_WRITE, sizeof(int)*n);
    cl::Buffer buffer_B3(context, CL_MEM_READ_WRITE, sizeof(int)*n);
    cl::Buffer buffer_C3(context, CL_MEM_READ_WRITE, sizeof(int)*n);
    cl::Buffer buffer_constants3(context, CL_MEM_READ_ONLY, sizeof(int) * 2);
    queue.enqueueWriteBuffer(buffer_A3, CL_TRUE, 0, sizeof(int)*n, A);
    queue.enqueueWriteBuffer(buffer_B3, CL_TRUE, 0, sizeof(int)*n, B);
    queue.enqueueWriteBuffer(buffer_constants3, CL_TRUE, 0, sizeof(int)*2, constants);

    add_single.setArg(0, buffer_A3);
    add_single.setArg(1, buffer_B3);
    add_single.setArg(2, buffer_C3);
    add_single.setArg(3, buffer_constants3);
    queue.enqueueNDRangeKernel(add_single, cl::NullRange, cl::NDRange(n), cl::NDRange(32));

    queue.enqueueReadBuffer(buffer_C3, CL_TRUE, 0, sizeof(int)*n, C);
    queue.enqueueBarrier();
    GPUtime3 = (std::clock() - start_time) / (double) CLOCKS_PER_SEC;

    // VERSION 4 ==========================================
    double GPUtime4;
    start_time = std::clock();

    cl::Buffer buffer_A4(context, CL_MEM_READ_WRITE, sizeof(int)*n);
    cl::Buffer buffer_B4(context, CL_MEM_READ_WRITE, sizeof(int)*n);
    cl::Buffer buffer_C4(context, CL_MEM_READ_WRITE, sizeof(int)*n);
    cl::Buffer buffer_constants4(context, CL_MEM_READ_ONLY, sizeof(int)*2);
    queue.enqueueWriteBuffer(buffer_A4, CL_TRUE, 0, sizeof(int)*n, A);
    queue.enqueueWriteBuffer(buffer_B4, CL_TRUE, 0, sizeof(int)*n, B);
    queue.enqueueWriteBuffer(buffer_constants4, CL_TRUE, 0, sizeof(int)*2, constants);

    add_single_overhead.setArg(0, buffer_A4);
    add_single_overhead.setArg(1, buffer_B4);
    add_single_overhead.setArg(2, buffer_C4);
    add_single_overhead.setArg(3, buffer_constants4);
    queue.enqueueNDRangeKernel(add_single_overhead, cl::NullRange, cl::NDRange(n), cl::NDRange(32));

    queue.enqueueReadBuffer(buffer_C4, CL_TRUE, 0, sizeof(int)*n, C);
    queue.enqueueBarrier();
    GPUtime4 = (std::clock() - start_time) / (double) CLOCKS_PER_SEC;

    // VERSION 5 ==========================================
    double GPUtime5;
    start_time = std::clock();

    cl::Buffer buffer_A5(context, CL_MEM_READ_WRITE, sizeof(int) * n);
    cl::Buffer buffer_B5(context, CL_MEM_READ_WRITE, sizeof(int) * n);
    cl::Buffer buffer_C5(context, CL_MEM_READ_WRITE, sizeof(int) * n);
    cl::Buffer buffer_constants5(context, CL_MEM_READ_ONLY, sizeof(int) * 2);
    queue.enqueueWriteBuffer(buffer_A5, CL_TRUE, 0, sizeof(int)*n, A);
    queue.enqueueWriteBuffer(buffer_B5, CL_TRUE, 0, sizeof(int)*n, B);
    queue.enqueueWriteBuffer(buffer_constants5, CL_TRUE, 0, sizeof(int)*2, constants);

    add_looped.setArg(0, buffer_A5);
    add_looped.setArg(1, buffer_B5);
    add_looped.setArg(2, buffer_C5);
    add_looped.setArg(3, buffer_constants5);
    queue.enqueueNDRangeKernel(add_looped, cl::NullRange, cl::NDRange(n), cl::NDRange(32));

    queue.enqueueReadBuffer(buffer_C5, CL_TRUE, 0, sizeof(int)*n, C);
    queue.enqueueBarrier();
    GPUtime5 = (std::clock() - start_time) / (double) CLOCKS_PER_SEC;
    
    // let's compare!
    const int NUM_VERSIONS = 5;
    double GPUtimes[NUM_VERSIONS] = {GPUtime1, GPUtime2, GPUtime3, GPUtime4, GPUtime5};
    if (verbose) {
        for (int i=0; i<NUM_VERSIONS; i++)
            compareResults(CPUtime, GPUtimes[i], i+1);
    } else {
        std::cout << CPUtime << ",";
        for (int i=0; i<NUM_VERSIONS-1; i++)
            std::cout << GPUtimes[i] << ",";
        std::cout << GPUtimes[NUM_VERSIONS-1] << std::endl;
    }
    return 0;
}


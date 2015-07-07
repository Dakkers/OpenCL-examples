#include <iostream>
#include <ctime>
#ifdef __APPLE__
    #include <OpenCL/cl.hpp>
#else
    #include <CL/cl.hpp>
#endif

#define NUM_GLOBAL_WITEMS 1024

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


void warmup(cl::Context &context, cl::CommandQueue &queue, 
            cl::Kernel &add, int A[], int B[], int n) {
    int C[n];
    // allocate space
    cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(int) * n);
    cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(int) * n);
    cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, sizeof(int) * n);

    // push write commands to queue
    queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(int)*n, A);
    queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(int)*n, B);

    // RUN ZE KERNEL
    add.setArg(1, buffer_B);
    add.setArg(0, buffer_A);
    add.setArg(2, buffer_C);
    for (int i=0; i<5; i++)
        queue.enqueueNDRangeKernel(add, cl::NullRange, cl::NDRange(NUM_GLOBAL_WITEMS), cl::NDRange(32));              
   
    queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(int)*n, C); 
    queue.finish(); 
}


int main(int argc, char* argv[]) {

    bool verbose;
    if (argc == 1 || std::strcmp(argv[1], "0") == 0)
        verbose = true;
    else
        verbose = false;
    
    const int n = 8*32*512;             // size of vectors
    const int k = 10000;                // number of loop iterations
    // const int NUM_GLOBAL_WITEMS = 1024; // number of threads

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
        "   void kernel add(global const int* v1, global const int* v2, global int* v3) {"
        "       int ID;"
        "       ID = get_global_id(0);"
        "       v3[ID] = v1[ID] + v2[ID];"
        "   }"
        ""
        "   void kernel add_looped_1(global const int* v1, global const int* v2, global int* v3, "
        "                          const int n, const int k) {"
        "       int ID, NUM_GLOBAL_WITEMS, ratio, start, stop;"
        "       ID = get_global_id(0);"
        "       NUM_GLOBAL_WITEMS = get_global_size(0);"
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
        "   void kernel add_looped_2(global const int* v1, global const int* v2, global int* v3,"
        "                            const int n, const int k) {"
        "       int ID, NUM_GLOBAL_WITEMS, step;"
        "       ID = get_global_id(0);"
        "       NUM_GLOBAL_WITEMS = get_global_size(0);"
        "       step = (n / NUM_GLOBAL_WITEMS);"
        ""
        "       int i,j;"
        "       for (i=0; i<k; i++) {"
        "           for (j=ID; j<n; j+=step)"
        "               v3[j] = v1[j] + v2[j];"
        "       }"
        "   }"
        ""    
        "   void kernel add_single(global const int* v1, global const int* v2, global int* v3, "
        "                          const int k) { "
        "       int ID = get_global_id(0);"
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
    cl::Kernel add          = cl::Kernel(program, "add");
    cl::Kernel add_looped_1 = cl::Kernel(program, "add_looped_1");
    cl::Kernel add_looped_2 = cl::Kernel(program, "add_looped_2");
    cl::Kernel add_single   = cl::Kernel(program, "add_single");

    // construct vectors
    int A[n], B[n], C[n];
    for (int i=0; i<n; i++) {
        A[i] = i;
        B[i] = n - i - 1;
    }

    // attempt at warm-up...
    warmup(context, queue, add, A, B, n);
    queue.finish();

    std::clock_t start_time;

    // VERSION 1 ==========================================
    // start timer
    double GPUtime1;
    start_time = std::clock();

    // allocate space
    cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(int) * n);
    cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(int) * n);
    cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, sizeof(int) * n);

    // push write commands to queue
    queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(int)*n, A);
    queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(int)*n, B);

    // RUN ZE KERNEL
    add_looped_1.setArg(0, buffer_A);
    add_looped_1.setArg(1, buffer_B);
    add_looped_1.setArg(2, buffer_C);
    add_looped_1.setArg(3, n);
    add_looped_1.setArg(4, k);
    queue.enqueueNDRangeKernel(add_looped_1, cl::NullRange,  // kernel, offset
            cl::NDRange(NUM_GLOBAL_WITEMS), // global number of work items
            cl::NDRange(32));               // local number (per group)

    // read result from GPU to here; including for the sake of timing
    queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(int)*n, C);
    queue.finish();
    GPUtime1 = (std::clock() - start_time) / (double) CLOCKS_PER_SEC;


    // VERSION 2 ==========================================
    double GPUtime2;

    cl::Buffer buffer_A2(context, CL_MEM_READ_WRITE, sizeof(int)*n);
    cl::Buffer buffer_B2(context, CL_MEM_READ_WRITE, sizeof(int)*n);
    cl::Buffer buffer_C2(context, CL_MEM_READ_WRITE, sizeof(int)*n);
    queue.enqueueWriteBuffer(buffer_A2, CL_TRUE, 0, sizeof(int)*n, A);
    queue.enqueueWriteBuffer(buffer_B2, CL_TRUE, 0, sizeof(int)*n, B);

    start_time = std::clock();
    add_looped_2.setArg(0, buffer_A2);
    add_looped_2.setArg(1, buffer_B2);
    add_looped_2.setArg(2, buffer_C2);
    add_looped_2.setArg(3, n);
    add_looped_2.setArg(4, k);
    
    queue.enqueueNDRangeKernel(add_looped_2, cl::NullRange, cl::NDRange(NUM_GLOBAL_WITEMS), cl::NDRange(32));
    queue.enqueueReadBuffer(buffer_C2, CL_TRUE, 0, sizeof(int)*n, C);
    queue.finish();
    GPUtime2 = (std::clock() - start_time) / (double) CLOCKS_PER_SEC;

    // let's compare!
    const int NUM_VERSIONS = 2;
    double GPUtimes[NUM_VERSIONS] = {GPUtime1, GPUtime2};
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


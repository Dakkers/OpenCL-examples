(NOTE: this content is from http://simpleopencl.blogspot.com/2013/06/tutorial-simple-start-with-opencl-and-c.html -- I am copying the content here so that it doesn't get lost to the sands of time.)

Saturday, June 1, 2013
Tutorial: Simple start with OpenCL and C++
To begin programming in OpenCL is always hard. Let's try with the basic example. We want to sum two arrays together.

At first you need to install the OpenCL libraries and other files. AMD has for CPU's and their GPU's  AMD APP: http://developer.amd.com/tools-and-sdks/heterogeneous-computing/amd-accelerated-parallel-processing-app-sdk/downloads/. Intel has their OpenCL libraries at http://software.intel.com/en-us/vcsource/tools/opencl-sdk. And Nvidia has everything at https://developer.nvidia.com/cuda-downloads. In some cases the graphic drivers already include all the files you need. I recommend that you continue with the next step and if anything will go wrong return to this step and install the needed OpenCL SDK toolkits.


We will program in C++11. To ease everything we will use OpenCL C++ binding 1.1 from www.khronos.org/registry/cl/api/1.1/cl.hpp . manual for this binding is available at www.khronos.org/registry/cl/specs/opencl-cplusplus-1.1.pdf. It might happen that cl.hpp is already installed at your computer. If not then simply download C++ binding to folder of your project. Don't forget to turn on the C++11. In case of QtCreator add next line into the .pro file:

```bash
QMAKE_CXXFLAGS += -std=c++0x
```

Also don't forget to use OpenCL library. In case of QtCreator add next line into the .pro file:

```bash
LIBS+= -lOpenCL
```

If you get any errors you need to adjust system variable to point to folder of OpenCL installation. You can also manually set path to OpenCL library path:

```bash
LIBS+= -Lpath_to_openCL_libraries
```

Or you can simply write hard-coded path to OpenCL library:

```bash
LIBS+=/usr/.../libOpenCL.so
```

Let's start with coding. We will create simple console program which will use OpenCL to sum two arrays like C=A+B. For our simple sample we will need only two headers:

```c
#include <iostream>
#include <CL/cl.hpp>
```

Everything else will happen inside main function. At start we need to get one of the OpenCL platforms. This is actually a driver you had previously installed. So platform can be from Nvidia, Intel, AMD....

```c
int main(){
    //get all platforms (drivers)
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if(all_platforms.size()==0){
        std::cout<<" No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Platform default_platform=all_platforms[0];
    std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";
```

Once we selected the first platform (default_platform) we will use it in the next steps. Now we need to get device of our platform. For example AMD's platform has support for multiple devices (CPU's and GPU's). We will now select the first device (default_device):

```c
//get default device of the default platform
std::vector<cl::Device> all_devices;
default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
if(all_devices.size()==0){
    std::cout<<" No devices found. Check OpenCL installation!\n";
    exit(1);
}
cl::Device default_device=all_devices[0];
std::cout<< "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";
```

Now we need to create a Context. Imagine the Context as the runtime link to the our device and platform:

```c
cl::Context context({default_device});
```

Next we need to create the program which we want to execute on our device:

```c
cl::Program::Sources sources;
```

Actual source of our program(kernel) is there:

```c
// kernel calculates for each element C=A+B
std::string kernel_code=
        "   void kernel simple_add(global const int* A, global const int* B, global int* C){       "
        "       C[get_global_id(0)]=A[get_global_id(0)]+B[get_global_id(0)];                 "
        "   }                                                                               ";
```

This code simply calculates C=A+B. As we want that one thread calculates sum of only one element, we use get_global_id(0). get_global_id(0) means get id of current thread. Id's can go from 0 to get_global_size(0) - 1. get_global_size(0) means number of threads. What is 0? 0 means first dimension. OpenCL supports running kernels on 1D, 2D and 3D problems. We will use 1D array! This means 1D problem.

Next we need our kernel sources to build. We also check for the errors at building:

```c
sources.push_back({kernel_code.c_str(),kernel_code.length()});

cl::Program program(context,sources);
if(program.build({default_device})!=CL_SUCCESS){
    std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device)<<"\n";
    exit(1);
}
```

For arrays A, B, C we need to allocate the space on the device:

```c
// create buffers on the device
cl::Buffer buffer_A(context,CL_MEM_READ_WRITE,sizeof(int)*10);
cl::Buffer buffer_B(context,CL_MEM_READ_WRITE,sizeof(int)*10);
cl::Buffer buffer_C(context,CL_MEM_READ_WRITE,sizeof(int)*10);
```

Arrays will have 10 element. We want to calculate sum of next arrays (A, B).

```c
int A[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
int B[] = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0};
```

We need to copy arrays from A and B to the device. This means that we will copy arrays from the host to the device. Host represents our main. At first we need to create a queue which is the queue to the commands we will send to the our device:

```c
//create queue to which we will push commands for the device.
cl::CommandQueue queue(context,default_device); Now we can copy data from arrays A and B to buffer_A and buffer_B which represent memory on the device:
//write arrays A and B to the device
queue.enqueueWriteBuffer(buffer_A,CL_TRUE,0,sizeof(int)*10,A);
queue.enqueueWriteBuffer(buffer_B,CL_TRUE,0,sizeof(int)*10,B);
```

Now we can run the kernel which in parallel sums A and B and writes to C. We do this with KernelFunctor which runs the kernel on the device. Take a look at the "simple_add" this is the name of our kernel we wrote before. You can see the number 10. This corresponds to number of threads we want to run (our array size is 10):

```c
cl::KernelFunctor simple_add(cl::Kernel(program,"simple_add"),queue,cl::NullRange,cl::NDRange(10),cl::NullRange); Here we actually set the arguments to kernel simple_add and run the kernel:
simple_add(buffer_A, buffer_B, buffer_C);
```

At the end we want to print memory C on our device. At first we need to transfer data from the device to our program (host):

```c
int C[10];
//read result C from the device to array C
queue.enqueueReadBuffer(buffer_C,CL_TRUE,0,sizeof(int)*10,C);

std::cout<<" result: \n";
for(int i=0;i<10;i++){
    std::cout<<C[i]<<" ";
}

return 0;
```

This is it. Complete code is there:

```c
#include <iostream>
#include <CL/cl.hpp>

int main(){
    //get all platforms (drivers)
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if(all_platforms.size()==0){
        std::cout<<" No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Platform default_platform=all_platforms[0];
    std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";

    //get default device of the default platform
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if(all_devices.size()==0){
        std::cout<<" No devices found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Device default_device=all_devices[0];
    std::cout<< "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";


    cl::Context context({default_device});

    cl::Program::Sources sources;

    // kernel calculates for each element C=A+B
    std::string kernel_code=
            "   void kernel simple_add(global const int* A, global const int* B, global int* C){       "
            "       C[get_global_id(0)]=A[get_global_id(0)]+B[get_global_id(0)];                 "
            "   }                                                                               ";
    sources.push_back({kernel_code.c_str(),kernel_code.length()});

    cl::Program program(context,sources);
    if(program.build({default_device})!=CL_SUCCESS){
        std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device)<<"\n";
        exit(1);
    }


    // create buffers on the device
    cl::Buffer buffer_A(context,CL_MEM_READ_WRITE,sizeof(int)*10);
    cl::Buffer buffer_B(context,CL_MEM_READ_WRITE,sizeof(int)*10);
    cl::Buffer buffer_C(context,CL_MEM_READ_WRITE,sizeof(int)*10);

    int A[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int B[] = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0};

    //create queue to which we will push commands for the device.
    cl::CommandQueue queue(context,default_device);

    //write arrays A and B to the device
    queue.enqueueWriteBuffer(buffer_A,CL_TRUE,0,sizeof(int)*10,A);
    queue.enqueueWriteBuffer(buffer_B,CL_TRUE,0,sizeof(int)*10,B);


    //run the kernel
    cl::KernelFunctor simple_add(cl::Kernel(program,"simple_add"),queue,cl::NullRange,cl::NDRange(10),cl::NullRange);
    simple_add(buffer_A,buffer_B,buffer_C);

    //alternative way to run the kernel
    /*cl::Kernel kernel_add=cl::Kernel(program,"simple_add");
    kernel_add.setArg(0,buffer_A);
    kernel_add.setArg(1,buffer_B);
    kernel_add.setArg(2,buffer_C);
    queue.enqueueNDRangeKernel(kernel_add,cl::NullRange,cl::NDRange(10),cl::NullRange);
    queue.finish();*/

    int C[10];
    //read result C from the device to array C
    queue.enqueueReadBuffer(buffer_C,CL_TRUE,0,sizeof(int)*10,C);

    std::cout<<" result: \n";
    for(int i=0;i<10;i++){
        std::cout<<C[i]<<" ";
    }

    return 0;
}
```

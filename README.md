# OpenCL basic examples
here is my feeble attempt at learning OpenCL, please don't make fun of me too much :hamburger:

## Configuration
This code uses OpenCL 1.1 on a NVIDIA GPU.

### Linux
(Only tested on Ubuntu). For NVIDIA GPUs, I've installed the following packages: `nvidia-346 nvidia-346-dev nvidia-346-uvm nvidia-libopencl1-346 nvidia-modprobe nvidia-opencl-icd-346 nvidia-settings`. Since the `opencl-headers` package in the main repository is for OpenCL 1.2, you can get the OpenCL 1.1 header files from [here](http://packages.ubuntu.com/precise/opencl-headers).

Then to compile the C++ code:

```
g++ -std=c++0x main.cpp -o main.out -lOpenCL
```

To compile the C code:

```
gcc main.c -o main.out -lOpenCL
```

### OS X
OpenCL is installed on OS X by default, but since this code uses the C++ bindings, you'll need to get that too. Get the [official C++ bindings from the OpenCL registr](https://www.khronos.org/registry/cl/api/1.1/cl.hpp) and copy it to the OpenCL framework directory, or do the following:

```
wget https://www.khronos.org/registry/cl/api/1.1/cl.hpp
sudo cp cl.hpp /System/Library/Frameworks/OpenCL.framework/Headers/
```

To compile:

```
clang++ -std=c++0x -framework OpenCL main.cpp -o main.out
```

## example 00
this example is based off of [this example](simpleopencl.blogspot.ca/2013/06/tutorial-simple-start-with-opencl-and-c.html) (example-ception), but it goes a bit further. In the blogspot example, two 10-element vectors are created and a thread is used for each pair of elements. In this example, 10 threads are spawned but two 100-element vectors are used, and it is shown how to split up a specific number of elements per thread.

## example 01
Measures the duration of adding two vectors. See the README in the folder for more details.

## example 02
Demonstrates that one array can be modified several times without having to re-read and re-write data to and from the GPU.

## TODO

- figure out how OpenCL manages memory (when are buffers cleared on the GPU?)
- figure out how to view the OpenCL assembly code if possible (is warp divergence happening?)

## Some Notes
From the [guide on programming OpenCL for NVIDIA](http://www.nvidia.com/content/cudazone/download/OpenCL/NVIDIA_OpenCL_ProgrammingGuide.pdf):

- **CUDA streaming multiprocessor** corresponds to an OpenCL compute unit
- **CUDA thread** corresponds to an OpenCL work-item
- **CUDA thread block** corresponds to an OpenCL work-group


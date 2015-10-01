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

For examples 04 and 05, you can run

```bash
make ex04  # executable is ./example04/bin/Example
make ex05  # executable is ./example05/bin/Example
make       # makes both!
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

### Windows
For some reason, the makefile didn't want to work for Windows. I have no idea why.

For example 04, run (inside the directory):

```
gcc -I/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v7.5/include -I/c/PATH/TO/CLFFT/include main.c -o main.exe -L/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v7.5/lib/x64 -lOpenCL -L/c/PATH/TO/CLFFT/lib64/import -lclFFT
```

where `PATH/TO/CLFFT` is the path to the clFFT library.

For example 05, run (inside the directory):

```
gcc -I/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v7.5/include -I/c/PATH/TO/CLFFT/include -I/c/PATH/TO/FFTW main.c -o main.exe -L/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v7.5/lib/x64 -lOpenCL -L/c/PATH/TO/CLFFT/lib64/import -lclFFT -L/c/PATH/TO/FFTW -lfftw3-3
```

where `PATH/TO/FFTW` is the path to the FFTW3 library.

## example 00
this example is based off of [this example](simpleopencl.blogspot.ca/2013/06/tutorial-simple-start-with-opencl-and-c.html) (example-ception), but it goes a bit further. In the blogspot example, two 10-element vectors are created and a thread is used for each pair of elements. In this example, 10 threads are spawned but two 100-element vectors are used, and it is shown how to split up a specific number of elements per thread.

## example 01
Measures the duration of adding two vectors. See the README in the folder for more details.

## example 02
Demonstrates that one array can be modified several times without having to re-read and re-write data to and from the GPU.

## example 03
A simple example using the `cl_khr_fp64` extension which allows for usage of doubles instead of floats.

## example 04
An example of the CLFFT library for an in-place complex-planar transform. There is also Python code to check the answer; FFTW code will be added later, probably.

- clFFT is required; installation instructions can be found inside example04/README.md
- for Python, numpy and scipy are required

## example 05
Another CLFFT example where an in-place real transform and an out-of-place real transform are performed. There's also FFTW code and Python code for checking the answer.

- clFFT is required; installation instructions can be found inside example04/README.md
- FFTW is required; installation is as simple as extracting FFTW's tar file, then running `./configure && sudo make && sudo make install`
- for Python, numpy and scipy are required

## Some Notes
From the [guide on programming OpenCL for NVIDIA](http://www.nvidia.com/content/cudazone/download/OpenCL/NVIDIA_OpenCL_ProgrammingGuide.pdf):

- **CUDA streaming multiprocessor** corresponds to an OpenCL compute unit
- **CUDA thread** corresponds to an OpenCL work-item
- **CUDA thread block** corresponds to an OpenCL work-group


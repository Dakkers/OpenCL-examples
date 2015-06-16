# OpenCL basic examples
here is my feeble attempt at learning OpenCL, please don't make fun of me too much :hamburger:

## Configuration
This currently runs on OS X, and I'm using local header files instead of global header files because I'm unfamiliar with C++. Deal with it. Run the following in a terminal to set up:

```
git clone git@github.com:SaintDako/OpenCL-examples.git
mkdir CL
curl https://www.khronos.org/registry/cl/api/1.2/cl.hpp -o CL/cl.hpp
```

## Compiling

```
clang++ -std=c++0x -framework OpenCL example01.cpp -o example01.out
```

To ignore the deprecation warnings, add the flag `-Wno-deprecated-declarations`.

## example 00
this example is based off of [this example](simpleopencl.blogspot.ca/2013/06/tutorial-simple-start-with-opencl-and-c.html) (example-ception), but it goes a bit further. In the blogspot example, two 10-element vectors are created and a thread is used for each pair of elements. In this example, 10 threads are spawned but two 100-element vectors are used, and it is shown how to split up a specific number of elements per thread.

## example 01
timing results of vectors added together on a CPU vs GPU. the size of the vectors and the number of times they are added together can be specified, as can the number of threads used in the GPU cases. currently implemented:

- CPU code
- GPU code equivalent
- GPU code where the buffers are created and destroyed each iteration (this is guaranteed to be slower, of course, but it would be nice to see the amount of overhead that occurs with inefficient copying...)
- GPU code where the number of threads spawned is equal to the size of the vector, and each thread adds one element

todo:

- GPU code where a work-group barrier is initiated after each thread is done its work (equivalent to the regular GPU code, but with one extra line...)

### results
so far, the third version (each thread doing only one element) is the fastest, and I'm trying to understand why.

## TODO

- figure out how OpenCL manages memory (when are buffers cleared on the GPU?)
- figure out how to view the OpenCL assembly code if possible (is warp divergence happening?)

## Some Notes
From the [guide on programming OpenCL for NVIDIA](http://www.nvidia.com/content/cudazone/download/OpenCL/NVIDIA_OpenCL_ProgrammingGuide.pdf):

- **CUDA streaming multiprocessor** corresponds to an OpenCL compute unit
- **CUDA thread** corresponds to an OpenCL work-item
- **CUDA thread block** corresponds to an OpenCL work-group


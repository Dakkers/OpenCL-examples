# OpenCL basic examples
here is my feeble attempt at learning OpenCL, please don't make fun of me too much :hamburger:

## Configuration
This currently runs on OS X, and I'm using local header files instead of global header files because I'm unfamiliar with C++. Deal with it. Run the following in a terminal to set up:

```
git clone git@github.com:SaintDako/OpenCL-examples.git
cd OpenCL-examples
mkdir CL
curl https://www.khronos.org/registry/cl/api/1.2/cl.hpp -o CL/cl.hpp
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


# OpenCL basic examples
here is my feeble attempt at learning OpenCL, please don't make fun of me too much :hamburger

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


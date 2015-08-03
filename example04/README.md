# example 04

## Installing CLFFT
After cloning the repository, run the following (in the top-level of the directory):

```
git clone https://github.com/clMathLibraries/clFFT.git
cd clFFT
mkdir build
cd build
cmake ../src
make
make install
```

and then

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`/clFFT/build/package/lib64
```

(`pwd` should give the path to the top-level of the repository!).

## Running it
In the top-level directory, run

```
make ex04
./example04/bin/Example
```

and it should print out a vector! :hamburger:

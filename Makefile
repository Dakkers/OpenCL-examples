CXX = gcc 

# clFFT lib & inc
CLFFT_LIB = -lOpenCL -L./usr/local/lib64 -lclFFT
CLFFT_INCLUDE = -I./clFFT/build/package/include

# standard math library
CXXFLAGS = -c $(CLFFT_INCLUDE)
LDFLAGS = -lm $(CLFFT_LIB) -lfftw3 -lm
EXE = Example 

all: ex04 ex05

ex04: example04/build/main.o
	@if [ ! -d "./example04/bin" ]; then mkdir ./example04/bin; fi
	$(CXX) $< $(LDFLAGS) -o example04/bin/$(EXE)

example04/build/main.o: example04/main.c
	@if [ ! -d "./example04/build" ]; then mkdir ./example04/build; fi
	$(CXX) $(CXXFLAGS) $< -o $@


ex05: example05/build/main.o
	@if [ ! -d "./example05/bin" ]; then mkdir ./example05/bin; fi
	$(CXX) $< $(LDFLAGS) -o example05/bin/$(EXE)

# create object file (compile without linking)
example05/build/main.o: example05/main.c
	@if [ ! -d "./example05/build" ]; then mkdir ./example05/build; fi
	$(CXX) $(CXXFLAGS) $< -o $@


# cleaning (remove executables and what not)
clean:
	$(RM) -r ./example04/build/ ./example04/bin/
	$(RM) -r ./example05/build/ ./example05/bin/

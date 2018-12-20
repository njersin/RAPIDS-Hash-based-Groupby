# Makefile for GPU GroupBy Project
# EE-5351 Fall 2018
NVCC        = nvcc
NVCC_FLAGS  = -I/usr/local/cuda/include -gencode=arch=compute_60,code=\"sm_60\" --relocatable-device-code true
CXX_FLAGS   = -std=c++11
ifdef dbg
	NVCC_FLAGS  += -g -G
else
	NVCC_FLAGS  += -O3
endif

LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	        = groupby
OBJ	        = main.o cpuGroupby.o hashfunction.o hashtable.o hashkernel.o

default: $(EXE)

main.o: main.cu cpuGroupby.h hashkernel.cuh
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS) $(CXX_FLAGS)

hashfunction.o: hashfunction.cu hashfunction.cuh
	$(NVCC) -c -o $@ hashfunction.cu $(NVCC_FLAGS)

hashtable.o: hashtable.cu hashtable.cuh
	$(NVCC) -c -o $@ hashtable.cu $(NVCC_FLAGS)

hashkernel.o: hashkernel.cu hashtable.cuh hashfunction.cuh hashkernel.cuh
	$(NVCC) -c -o $@ hashkernel.cu $(NVCC_FLAGS) $(CXX_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS) $(NVCC_FLAGS)

clean:
	rm -rf *.o $(EXE)

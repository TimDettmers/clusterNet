CC = nvcc
MPI_DIR=/usr/mpi/openmpi-1.7.4
TOP := $(dir $(CURDIR)/$(word $(words $(MAKEFILE_LIST)),$(MAKEFILE_LIST)))
TESTS := tests/testSuite.cu $(wildcard tests/*_test.cu) 
BUILD := $(subst .cu,.out,$(subst tests/, build/, $(wildcard tests/*_test.cu)))
NODES=tim@10.0.0.2
HOSTFILE=/home/tim/cluster
SCR := $(wildcard source/*.cu)
INCLUDE = -I $(MPI_DIR)/include -I $(TOP)source -I $(TOP)tests -I /usr/local/cuda-5.5/include
LIB = -L $(MPI_DIR)/lib -L /usr/local/cuda-5.5/lib64
CFLAGS = -gencode arch=compute_35,code=sm_35 -lcublas -lcurand -lmpi_cxx -lmpi $(LIB) $(INCLUDE) 
LINK = source/util.cu source/clusterKernels.cu source/basicOps.cu source/clusterNet.cpp 

EXECSRC = build/clusterNet.out
EXECTEST = build/testSuite.out

all : $(EXECSRC) $(EXECTEST) 
	
$(EXECSRC) : $(SCR) source/clusterNet.cpp
	$(CC) $^ -o $@ $(CFLAGS)

$(EXECTEST): $(SCR) $(TESTS) source/clusterNet.cpp    
	$(CC) $(TESTS) $(LINK) -o $@ $(CFLAGS)

test:
	scp $(TOP)$(EXECTEST) $(NODES):$(TOP)build/;	
	$(MPI_DIR)/bin/mpirun -np 2 -hostfile $(HOSTFILE) $(TOP)$(EXECTEST) 

run:
	scp $(TOP)$(EXECSRC) $(NODES):$(TOP)build/;
	$(MPI_DIR)/bin/mpirun -np 2 -hostfile $(HOSTFILE) $(TOP)$(EXECSRC) 

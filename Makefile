CC = nvcc
MPI_DIR=/usr/mpi/openmpi-1.7.4
TOP := $(dir $(CURDIR)/$(word $(words $(MAKEFILE_LIST)),$(MAKEFILE_LIST)))
TESTS := $(wildcard tests/*_test.cu)
BUILD := $(subst .cu,.out,$(subst tests/, build/, $(wildcard tests/*_test.cu)))
NODES=tim@10.0.0.2
HOSTFILE=/home/tim/cluster
SCR := $(wildcard source/*.cu)
INCLUDE = -I $(MPI_DIR)/include -I $(TOP)source
LIB = -L $(MPI_DIR)/lib -L /usr/local/cuda-5.5/lib64
CFLAGS = -gencode arch=compute_35,code=sm_35 -lcublas -lmpi $(LIB) $(INCLUDE)
LINK = source/util.cu source/clusterKernels.cu source/basicOps.cu


EXECSRC = build/clusterNet.out


$(EXECSRC): $(SCR)	     
	$(CC) $^ -o $@ $(CFLAGS)

compile_tests:     
	$(foreach T,$(TESTS),$(CC) $(LINK) $(T) -o $(subst .cu,.out,$(subst tests/, build/, $(T))) $(CFLAGS);)

test:
	$(foreach B, $(BUILD), 	scp $(TOP)$(B) $(NODES):$(TOP)build/;)
	$(foreach B, $(BUILD), $(MPI_DIR)/bin/mpirun -np 2 -hostfile $(HOSTFILE) $(B);)

run:
	echo $(TOP)
	scp $(TOP)build/clusterNet.out $(NODES):$(TOP)build/;
	$(MPI_DIR)/bin/mpirun -np 2 -hostfile $(HOSTFILE) $(TOP)build/clusterNet.out   

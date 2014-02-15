CC=nvcc
MPI_DIR=/usr/mpi/openmpi-1.7.4
SELF_DIR := $(dir $(lastword $(MAKEFILE_LIST)))
TESTS := $(wildcard tests/*_test.cu)
BUILD := $(subst .cu,.out,$(subst tests/, build/, $(wildcard tests/*_test.cu)))
NODES=tim@10.0.0.2
HOSTFILE=/home/tim/cluster
GIT_DIR=/home/tim/git

compile: $(TESTS)
	echo "Compiling all tests...";
	$(foreach T,$(TESTS),nvcc -lmpi $(T) -o $(subst .cu,.out,$(subst tests/, build/, $(T))) -I $(MPI_DIR)/include -L $(MPI_DIR)/lib -gencode arch=compute_35,code=sm_35;)

test:
	scp -r $(GIT_DIR)/clusterNet/build/ $(NODES):$(GIT_DIR)/clusterNet/build/
	$(foreach B, $(BUILD), $(MPI_DIR)/bin/mpirun -np 2 -hostfile $(HOSTFILE) $(B));

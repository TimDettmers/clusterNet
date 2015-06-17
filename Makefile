CC = nvcc
MPI_DIR=/usr/local/openmpi
#MPI_DIR=/usr/mpi/openmpi-1.8.1
HDF5_DIR = /home/tim/apps/hdf5-1.8.14/hdf5
SZIP_DIR = /home/tim/apps/szip-2.1/szip/
TOP := $(dir $(CURDIR)/$(word $(words $(MAKEFILE_LIST)),$(MAKEFILE_LIST)))
TESTS := tests/testSuite.cu $(wildcard tests/*_test.cu) 
NODES=tim@10.0.0.2
HOSTFILE=/home/tim/cluster_one_node
#HOSTFILE=/home/tim/cluster_other_node
#HOSTFILE=/home/tim/cluster
SCR := $(wildcard source/*.cu) $(wildcard source/*.cpp)
INCLUDE = -I $(MPI_DIR)/include -I $(TOP)source -I $(TOP)tests -I /usr/local/cuda/include -I $(HDF5_DIR)include -I $(SZIP_DIR)include
LIB = -L $(MPI_DIR)/lib -L /usr/local/cuda/lib64 -L $(HDF5_DIR)lib -L $(SZIP_DIR)lib
CFLAGS = -gencode arch=compute_35,code=sm_35 -lcusparse -lcublas -lcurand -lmpi_cxx -lmpi -lhdf5 -lhdf5_hl -lz $(LIB) $(INCLUDE) 
LINK = source/util.cu source/clusterKernels.cu source/basicOps.cu $(wildcard source/*.cpp)

EXECSRC = build/clusterNet.out
EXECTEST = build/testSuite.out

all : $(EXECSRC) #$(EXECTEST) 
	
$(EXECSRC) : $(SCR) 
	$(CC) $^ -o $@ $(CFLAGS)

$(EXECTEST): $(SCR) $(TESTS)    
	$(CC) $(TESTS) $(LINK) -o $@ $(CFLAGS)

test:
	#scp $(TOP)$(EXECTEST) $(NODES):$(TOP)build/;	
	$(MPI_DIR)/bin/mpirun -x LD_LIBRARY_PATH -np 2 $(TOP)$(EXECTEST)  

run:
	#scp $(TOP)$(EXECSRC) $(NODES):$(TOP)build/;
	$(MPI_DIR)/bin/mpirun -x LD_LIBRARY_PATH -np 2 $(TOP)$(EXECSRC)

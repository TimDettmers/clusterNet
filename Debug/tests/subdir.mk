################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../tests/basicOps_test.cu \
../tests/batchAllocator_test.cu \
../tests/clusterNet_test.cu \
../tests/miniMNIST_test.cu \
../tests/testSuite.cu \
../tests/util_test.cu 

CU_DEPS += \
./tests/basicOps_test.d \
./tests/batchAllocator_test.d \
./tests/clusterNet_test.d \
./tests/miniMNIST_test.d \
./tests/testSuite.d \
./tests/util_test.d 

OBJS += \
./tests/basicOps_test.o \
./tests/batchAllocator_test.o \
./tests/clusterNet_test.o \
./tests/miniMNIST_test.o \
./tests/testSuite.o \
./tests/util_test.o 


# Each subdirectory must supply rules for building sources it contributes
tests/%.o: ../tests/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-5.5/bin/nvcc -I/usr/mpi/openmpi-1.7.4/include -I/home/tim/git/clusterNet/source/ -I/home/tim/git/clusterNet/tests/ -I/usr/local/cuda/include -G -g -O0 -gencode arch=compute_35,code=sm_35 -odir "tests" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-5.5/bin/nvcc --compile -G -I/usr/mpi/openmpi-1.7.4/include -I/home/tim/git/clusterNet/source/ -I/home/tim/git/clusterNet/tests/ -I/usr/local/cuda/include -O0 -g -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '



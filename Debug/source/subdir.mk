################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../source/DeepNeuralNetwork.cpp \
../source/Layer.cpp \
../source/WikiMaxoutNet.cpp \
../source/WikiMaxoutNet_PCIe.cpp \
../source/WikiMaxoutNet_PCIe2.cpp \
../source/WikiNetDist.cpp \
../source/batchAllocator.cpp \
../source/clusterNet.cpp 

CU_SRCS += \
../source/basicOps.cu \
../source/clusterKernels.cu \
../source/test.cu \
../source/util.cu 

CU_DEPS += \
./source/basicOps.d \
./source/clusterKernels.d \
./source/test.d \
./source/util.d 

OBJS += \
./source/DeepNeuralNetwork.o \
./source/Layer.o \
./source/WikiMaxoutNet.o \
./source/WikiMaxoutNet_PCIe.o \
./source/WikiMaxoutNet_PCIe2.o \
./source/WikiNetDist.o \
./source/basicOps.o \
./source/batchAllocator.o \
./source/clusterKernels.o \
./source/clusterNet.o \
./source/test.o \
./source/util.o 

CPP_DEPS += \
./source/DeepNeuralNetwork.d \
./source/Layer.d \
./source/WikiMaxoutNet.d \
./source/WikiMaxoutNet_PCIe.d \
./source/WikiMaxoutNet_PCIe2.d \
./source/WikiNetDist.d \
./source/batchAllocator.d \
./source/clusterNet.d 


# Each subdirectory must supply rules for building sources it contributes
source/%.o: ../source/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-5.5/bin/nvcc -I/usr/mpi/openmpi-1.7.4/include -I/home/tim/git/clusterNet/source/ -I/home/tim/git/clusterNet/tests/ -I/usr/local/cuda/include -G -g -O0 -gencode arch=compute_35,code=sm_35 -odir "source" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-5.5/bin/nvcc -I/usr/mpi/openmpi-1.7.4/include -I/home/tim/git/clusterNet/source/ -I/home/tim/git/clusterNet/tests/ -I/usr/local/cuda/include -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

source/%.o: ../source/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-5.5/bin/nvcc -I/usr/mpi/openmpi-1.7.4/include -I/home/tim/git/clusterNet/source/ -I/home/tim/git/clusterNet/tests/ -I/usr/local/cuda/include -G -g -O0 -gencode arch=compute_35,code=sm_35 -odir "source" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-5.5/bin/nvcc --compile -G -I/usr/mpi/openmpi-1.7.4/include -I/home/tim/git/clusterNet/source/ -I/home/tim/git/clusterNet/tests/ -I/usr/local/cuda/include -O0 -g -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '



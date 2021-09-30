#compilers
CC=nvcc

#GLOBAL_PARAMETERS
VALUE_TYPE = double

#CUDA_PARAMETERS
NVCC_FLAGS = -O3 -w -m64 -Xptxas -dlcm=cg -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_61,code=compute_61

#ENVIRONMENT_PARAMETERS
CUDA_INSTALL_PATH = /usr/local/cuda

#includes
INCLUDES = -I$(CUDA_INSTALL_PATH)/include  

#libs
#CLANG_LIBS = -stdlib=libstdc++ -lstdc++
CUDA_LIBS = -L$(CUDA_INSTALL_PATH)/lib64  -lcudart
LIBS = $(CUDA_LIBS)

#options
#OPTIONS = -std=c99

TARGET_EXEC:=sptrsv



BUILD_DIR:=./build
all:$(BUILD_DIR)/$(TARGET_EXEC)
SRC_DIRS:=

SRCS:=$(shell find $(SRC_DIRS)-name *.cu)

OBJS:=$(SRCS:%=$(BUILD_DIR)/%.o)


DEPS:=$(OBJS:.o=.d)

$(BUILD_DIR)/$(TARGET_EXEC): $(OBJS)
	$(CC) $(OBJS) -o $@	$(LDFLAGS)

$(BUILD_DIR)/%.cu.o: %.cu
	mkdir -p $(dir $@)
	$(CC) $(NVCC_FLAGS) -c $< -o $@ $(INCLUDES) $(LIBS) $(OPTIONS) -D VALUE_TYPE=$(VALUE_TYPE)

clean:
	rm -rf $(BUILD_DIR)

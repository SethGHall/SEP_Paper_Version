#! CUDA
find_package(CUDA 10.1 REQUIRED)
message(STATUS ">>> Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")

set(
	CUDA_NVCC_FLAGS
	${CUDA_NVCC_FLAGS};
	-std=c++11 -rdc=true -arch=sm_61 -Xptxas -O3 -restrict -D_FORCE_INLINES -D_DEBUG -lineinfo
)

message(STATUS ">>> Cmake current directory: ${CMAKE_CURRENT_SOURCE_DIR}")

#! C based YAML Parser - https://github.com/pantoniou/libfyaml
find_library(fyaml_loc NAMES libfyaml-0.5.a)
message(STATUS ">>> Location of C YAML parser lib: ${fyaml_loc}")
add_library(fyaml-0.5 STATIC IMPORTED)
set_target_properties(fyaml-0.5 PROPERTIES IMPORTED_LOCATION ${fyaml_loc})



cuda_add_executable(
	imaging ../main.cpp 
	src/controller.cu
	src/common.cu
	src/gridder.cu
	src/msmfs_gridder.cu
	src/gains.cu
	src/deconvolution.cu
	src/direct_fourier_transform.cu
	src/timer.cu
	# src/system_test.cu
	src/imaging.cu
	src/wprojection.cu
	src/fft.cu
	src/complex.cu
	src/weighting.cu
	src/nifty.cu
	src/msmfs_nifty.cu
	src/fits_image_writer.cu
	src/restorer.cu
)
set_target_properties(imaging PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(imaging m fyaml-0.5 cudart cusolver cufft pthread)

target_include_directories(imaging PUBLIC .)

#! System test project
# project(system_test C CXX CUDA)

# cuda_add_executable(
#	system_test ../test.cpp 
#	src/controller.cu
#	src/common.cu
#	src/gridder.cu
#	src/gains.cu
#	src/deconvolution.cu
#	src/direct_fourier_transform.cu
#	src/timer.cu
#	src/system_test.cu
#	src/imaging.cu
#	src/wprojection.cu
#	src/fft.cu
#	src/nifty.cu
# )

# target_link_libraries(system_test m fyaml-0.5 cudart cusolver cufft pthread)

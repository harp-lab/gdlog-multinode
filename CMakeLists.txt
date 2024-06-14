cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(gdlog LANGUAGES CXX CUDA)

find_package(MPI REQUIRED)

option(POLARIS_RDMA "Turn on GPU direct on polaris" OFF)
option(FORCE_GPU_DIRECT "Force GPU direct to be on" OFF)
option(USE_64BIT_DOMAIN "Use u64 as builtin type for column domain" OFF)

if (USE_64BIT_DOMAIN)
    add_definitions(-DUSE_64_BIT_TUPLE)
endif()

# find_package(CUDA REQUIRED)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")

if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/gdlog_RAPIDS.cmake)
file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-24.06/RAPIDS.cmake
${CMAKE_CURRENT_BINARY_DIR}/gdlog_RAPIDS.cmake)
endif()

include(${CMAKE_CURRENT_BINARY_DIR}/gdlog_RAPIDS.cmake)

include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

rapids_cpm_init()

function(rapids_cpm_cuco)
list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cpm.cuco")

set(options)
set(one_value INSTALL_EXPORT_SET)
set(multi_value)
cmake_parse_arguments(_RAPIDS "${options}" "${one_value}" "${multi_value}" ${ARGN})

# Fix up _RAPIDS_UNPARSED_ARGUMENTS to have INSTALL_EXPORT_SET as this is need for rapids_cpm_find
set(to_install OFF)

if(_RAPIDS_INSTALL_EXPORT_SET)
list(APPEND _RAPIDS_UNPARSED_ARGUMENTS INSTALL_EXPORT_SET ${_RAPIDS_INSTALL_EXPORT_SET})
set(to_install ON)
endif()

include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
rapids_cpm_package_details(cuco version repository tag shallow exclude)

set(to_exclude OFF)

if(NOT to_install OR exclude)
set(to_exclude ON)
endif()

include("${rapids-cmake-dir}/cpm/detail/generate_patch_command.cmake")
rapids_cpm_generate_patch_command(cuco ${version} patch_command)

include("${rapids-cmake-dir}/cpm/find.cmake")
rapids_cpm_find(cuco ${version} ${_RAPIDS_UNPARSED_ARGUMENTS}
GLOBAL_TARGETS cuco::cuco
CPM_ARGS
GIT_REPOSITORY ${repository}
GIT_TAG ${tag}
GIT_SHALLOW ${shallow} ${patch_command}
EXCLUDE_FROM_ALL ${to_exclude}
OPTIONS "BUILD_TESTS OFF" "BUILD_BENCHMARKS OFF" "BUILD_EXAMPLES OFF"
"INSTALL_CUCO ${to_install}")

include("${rapids-cmake-dir}/cpm/detail/display_patch_status.cmake")
rapids_cpm_display_patch_status(cuco)

# Propagate up variables that CPMFindPackage provide
set(cuco_SOURCE_DIR "${cuco_SOURCE_DIR}" PARENT_SCOPE)
set(cuco_BINARY_DIR "${cuco_BINARY_DIR}" PARENT_SCOPE)
set(cuco_ADDED "${cuco_ADDED}" PARENT_SCOPE)
set(cuco_VERSION ${version} PARENT_SCOPE)
endfunction()

rapids_cpm_cuco()
function(rapids_cpm_rmm)
list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cpm.rmm")

set(options)
set(one_value INSTALL_EXPORT_SET)
set(multi_value)
cmake_parse_arguments(_RAPIDS "${options}" "${one_value}" "${multi_value}" ${ARGN})

# Fix up RAPIDS_UNPARSED_ARGUMENTS to have EXPORT_SETS as this is need for rapids_cpm_find
if(_RAPIDS_INSTALL_EXPORT_SET)
list(APPEND _RAPIDS_UNPARSED_ARGUMENTS INSTALL_EXPORT_SET ${_RAPIDS_INSTALL_EXPORT_SET})
endif()

include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
rapids_cpm_package_details(rmm version repository tag shallow exclude)
set(to_exclude OFF)

if(NOT _RAPIDS_INSTALL_EXPORT_SET OR exclude)
set(to_exclude ON)
endif()

include("${rapids-cmake-dir}/cpm/detail/generate_patch_command.cmake")
rapids_cpm_generate_patch_command(rmm ${version} patch_command)

include("${rapids-cmake-dir}/cpm/find.cmake")
rapids_cpm_find(rmm ${version} ${ARGN} ${_RAPIDS_UNPARSED_ARGUMENTS}
GLOBAL_TARGETS rmm::rmm
CPM_ARGS
GIT_REPOSITORY ${repository}
GIT_TAG ${tag}
GIT_SHALLOW ${shallow} ${patch_command}
EXCLUDE_FROM_ALL ${to_exclude}
OPTIONS "BUILD_TESTS OFF" "BUILD_BENCHMARKS OFF")

include("${rapids-cmake-dir}/cpm/detail/display_patch_status.cmake")
rapids_cpm_display_patch_status(rmm)

# Propagate up variables that CPMFindPackage provide
set(rmm_SOURCE_DIR "${rmm_SOURCE_DIR}" PARENT_SCOPE)
set(rmm_BINARY_DIR "${rmm_BINARY_DIR}" PARENT_SCOPE)
set(rmm_ADDED "${rmm_ADDED}" PARENT_SCOPE)
set(rmm_VERSION ${version} PARENT_SCOPE)

# rmm creates the correct namespace aliases
endfunction()

rapids_cpm_rmm()

# using test
include(CTest)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
    add_subdirectory(unittest)
endif()

enable_testing()

set(gpu_ra_src "${PROJECT_SOURCE_DIR}/src")
set(gpu_ra_include "${PROJECT_SOURCE_DIR}/include")
set(test_dir "${PROJECT_SOURCE_DIR}/test")

file(GLOB source_file_gpu_ra
    "${gpu_ra_include}/exception.cuh"
    "${gpu_ra_include}/dynamic_dispatch.h"
    "${gpu_ra_include}/print.cuh"
    "${gpu_ra_include}/relation.cuh"
    "${gpu_ra_include}/relational_algebra.cuh"
    "${gpu_ra_include}/timer.cuh"
    "${gpu_ra_include}/tuple.cuh"
    "${gpu_ra_include}/lie.cuh"
    "${gpu_ra_include}/comm.cuh"
    "${gpu_ra_include}/rt_incl.h"
    "${gpu_ra_src}/tuple.cu"
    "${gpu_ra_src}/hashtrie.cu"
    "${gpu_ra_src}/print.cu"
    "${gpu_ra_src}/relation.cu"
    "${gpu_ra_src}/lie.cu"
    "${gpu_ra_src}/ra/copy.cu"
    "${gpu_ra_src}/ra/acopy.cu"
    "${gpu_ra_src}/ra/join.cu"
    "${gpu_ra_src}/ra/filter.cu"
    "${gpu_ra_src}/ra/arithm.cu"
    "${gpu_ra_src}/ra/negation.cu"
    "${gpu_ra_src}/ra/cartesian.cu"
    "${gpu_ra_src}/ra/union.cu"
    "${gpu_ra_src}/broadcast.cu"
    "${gpu_ra_src}/tokenizer.cpp"
    "${gpu_ra_src}/ast.cpp"
    "${gpu_ra_src}/comm.cu"
    "${gpu_ra_src}/rt.cpp"
)

file(GLOB source_file_tc
    "${test_dir}/tc.cu"
)

file(GLOB source_path_lenght
    "${test_dir}/path_length.cu"
)

file(GLOB source_cspa
    "${test_dir}/cspa2.cu"
)

file(GLOB source_file_datastructure
    "${test_dir}/datastructure.cu"
)

file(GLOB source_file_sg
    "${test_dir}/sg.cu"
)

file(GLOB source_file_test
    "${test_dir}/sort.cu"
)

file(GLOB source_file_merge
    "${test_dir}/merge.cu"
)

add_library(gpu_ra "${source_file_gpu_ra}")
target_compile_features(gpu_ra PUBLIC cxx_std_17)

if(POLARIS_RDMA)
    add_library(mpi_gtl SHARED IMPORTED)
    set_property(TARGET mpi_gtl PROPERTY
        IMPORTED_LOCATION /opt/cray/pe/mpich/8.1.25/gtl/lib/libmpi_gtl_cuda.so)
    target_link_libraries(gpu_ra MPI::MPI_CXX rmm::rmm mpi_gtl)
    target_compile_definitions(gpu_ra PUBLIC DEFAULT_GPU_RDMA=1)
else()
    if (FORCE_GPU_DIRECT)
        target_compile_definitions(gpu_ra PUBLIC DEFAULT_GPU_RDMA=1)
    endif()
    target_link_libraries(gpu_ra MPI::MPI_CXX rmm::rmm cuco)
endif()

set_target_properties(gpu_ra PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES "80")
target_compile_options(gpu_ra PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
    -lineinfo
    --expt-relaxed-constexpr
    --default-stream per-thread
    --extended-lambda>)

add_executable(TC ${source_file_tc})
target_link_libraries(TC gpu_ra)
target_compile_features(TC PUBLIC cxx_std_17)
target_compile_options(TC PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
    -lineinfo
    --expt-relaxed-constexpr
    --default-stream per-thread
    --extended-lambda>)
set_target_properties(TC PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES "80")

# add_executable(CSPA2 ${source_cspa})
# target_link_libraries(CSPA2 gpu_ra)
# target_compile_features(CSPA2 PUBLIC cxx_std_17)
# set_target_properties(CSPA2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES "80")

# add_executable(SG ${source_file_sg})
# target_link_libraries(SG gpu_ra)
# target_compile_features(SG PUBLIC cxx_std_17)
# set_target_properties(SG PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES "80")


# file(GLOB source_file_block_overlap
#     "${test_dir}/block_overlap.cu"
# )
# add_executable(BLOCK_OVERLAP ${source_file_block_overlap})
# target_link_libraries(BLOCK_OVERLAP gpu_ra)
# target_compile_features(BLOCK_OVERLAP PUBLIC cxx_std_17)
# set_target_properties(BLOCK_OVERLAP PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES "80")

# file(GLOB source_file_value_analysis
#     "${test_dir}/value_analysis.cu"
# )
# add_executable(VALUE_ANALYSIS ${source_file_value_analysis})
# target_link_libraries(VALUE_ANALYSIS gpu_ra)
# target_compile_features(VALUE_ANALYSIS PUBLIC cxx_std_17)
# set_target_properties(VALUE_ANALYSIS PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES "80")

file(GLOB source_file_value_analysis_test
    "${test_dir}/value_analysis_debug.cu"
)
add_executable(VALUE_ANALYSIS_TEST ${source_file_value_analysis_test})
target_link_libraries(VALUE_ANALYSIS_TEST gpu_ra)
target_compile_features(VALUE_ANALYSIS_TEST PUBLIC cxx_std_17)
set_target_properties(VALUE_ANALYSIS_TEST PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES "80")




project(unittest LANGUAGES CXX CUDA)

option(USE_64BIT_DOMAIN "Use u64 as builtin type for column domain" OFF)

# add_executable(COMPILER_TEST src/compiler_test.cpp)
# target_include_directories(COMPILER_TEST PRIVATE ${CMAKE_SOURCE_DIR}/include)
# set_target_properties(COMPILER_TEST PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES "80")
# target_compile_options(COMPILER_TEST PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
#     -lineinfo
#     --expt-relaxed-constexpr
#     --default-stream per-thread
#     --expt-extended-lambda>)
# target_link_libraries(COMPILER_TEST PRIVATE gpu_ra)
# target_compile_features(COMPILER_TEST PRIVATE cxx_std_20)

# add_test(NAME COMPILER_TEST COMMAND COMPILER_TEST)

add_executable(RA_TEST src/ra_test.cu)
target_include_directories(RA_TEST PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(RA_TEST PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES "80")
target_compile_options(RA_TEST PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
    -lineinfo
    --expt-relaxed-constexpr
    --default-stream per-thread
    --expt-extended-lambda>)
target_link_libraries(RA_TEST PRIVATE gpu_ra)
target_compile_features(RA_TEST PRIVATE cxx_std_17)

add_test(NAME RA_TEST COMMAND RA_TEST)


add_executable(COMM_TEST src/comm_test.cu)
target_include_directories(COMM_TEST PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(COMM_TEST PRIVATE gpu_ra)
target_compile_features(COMM_TEST PRIVATE cxx_std_17)
set_target_properties(COMM_TEST PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES "80")
target_compile_options(COMM_TEST PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
    -lineinfo
    --expt-relaxed-constexpr
    --default-stream per-thread
    --expt-extended-lambda>)

add_test(NAME COMM_TEST COMMAND COMM_TEST)

# add TC test for TC
add_executable(TC_TEST src/tc_test.cu)
target_include_directories(TC_TEST PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(TC_TEST PRIVATE gpu_ra)
target_compile_features(TC_TEST PRIVATE cxx_std_17)
set_target_properties(TC_TEST PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES "80")
target_compile_options(TC_TEST PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
    -lineinfo
    --expt-relaxed-constexpr
    --default-stream per-thread
    --expt-extended-lambda>)

add_test(NAME TC_TEST COMMAND TC_TEST)

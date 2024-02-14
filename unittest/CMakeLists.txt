
project(unittest LANGUAGES CXX CUDA)

add_executable(COMPILER_TEST src/compiler_test.cpp)

target_include_directories(COMPILER_TEST PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(COMPILER_TEST PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES "80;89;90")
target_compile_options(COMPILER_TEST PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
    -lineinfo
    --expt-relaxed-constexpr
    --default-stream per-thread
    --expt-extended-lambda>)
target_link_libraries(COMPILER_TEST PRIVATE gpu_ra)
target_compile_features(COMPILER_TEST PRIVATE cxx_std_20)

add_test(NAME COMPILER_TEST COMMAND COMPILER_TEST)


add_executable(RA_TEST src/ra_test.cu)
target_include_directories(RA_TEST PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(RA_TEST PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES "80;89;90")
target_compile_options(RA_TEST PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
    -lineinfo
    --expt-relaxed-constexpr
    --default-stream per-thread
    --expt-extended-lambda>)
target_link_libraries(RA_TEST PRIVATE gpu_ra)
target_compile_features(RA_TEST PRIVATE cxx_std_20)

add_test(NAME RA_TEST COMMAND RA_TEST)

#include <cstdint>
#include <cuda/std/chrono>
#include <iostream>
#include <vector>
// thrust use TBB
// #define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_TBB

#include "../include/hashtrie.cuh"

#include "../include/exception.cuh"
#include "../include/lie.cuh"
#include "../include/print.cuh"
#include "../include/timer.cuh"

#include <execinfo.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// use librmm
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

void handler(int sig) {
    void *array[10];

    // get void*'s for all entries on the stack
    size_t size = backtrace(array, 10);
    char **strs = backtrace_symbols(array, size);
    for (int i = 0; i < size; i++) {
        printf("%s\n", strs[i]);
    }

    // print out all the frames to stderr
    fprintf(stderr, "Error: signal %d:\n", sig);
    backtrace_symbols_fd(array, size, STDERR_FILENO);
    exit(1);
}

const uint32_t TEST_CASE_SIZE = 10 * 10000;
const uint32_t REPEAT = 50;
const uint32_t TEST_ARIRTY = 2;

void generate_random_data(thrust::host_vector<uint32_t> &data_column1) {
    for (int i = 0; i < data_column1.size(); i++) {
        data_column1[i] = rand() % TEST_CASE_SIZE;
    }
}

void test_load(thrust::host_vector<thrust::host_vector<uint32_t>> &data_columns,
               hisa::hisa_cpu &hashtrie_cpu) {
    hashtrie_cpu.load_vectical(data_columns);
    hashtrie_cpu.deduplicate();
    hashtrie_cpu.build_index();
}

void test_load_hisa(thrust::host_vector<uint32_t> &data_columns,
                    Relation *rel) {
    int device_id;
    int number_of_sm;
    cudaGetDevice(&device_id);
    cudaDeviceGetAttribute(&number_of_sm, cudaDevAttrMultiProcessorCount,
                           device_id);
    int block_size, grid_size;
    block_size = 512;
    grid_size = 32 * number_of_sm;
    uint32_t rows = data_columns.size() / TEST_ARIRTY;
    load_relation(rel, "rel", TEST_ARIRTY, data_columns.data(), rows,
                  TEST_ARIRTY - 1, 0, grid_size, block_size);
}

void test_load_multi(
    thrust::host_vector<thrust::host_vector<uint32_t>> &data_columns,
    hisa::multi_hisa &h) {

    h.init_load_vectical(data_columns, data_columns[0].size());
    h.deduplicate();
    h.persist_newt();
}

void raw_vertical_to_horizontal(
    thrust::host_vector<thrust::host_vector<uint32_t>> &data_columns,
    thrust::host_vector<uint32_t> &data_columns_horizontal) {
    auto total_size = data_columns.size() * data_columns[0].size();
    for (int i = 0; i < data_columns[0].size(); i++) {
        for (int j = 0; j < data_columns.size(); j++) {
            data_columns_horizontal.push_back(data_columns[j][i]);
        }
    }
}

void testcase_deduplicate() {
    thrust::host_vector<thrust::host_vector<hisa::internal_data_type>> test_raw;
    test_raw.push_back({{1, 6, 3, 9, 1, 2, 3, 8}});
    test_raw.push_back({{3, 2, 3, 9, 3, 7, 3, 1}});
    // 3 8 1 1 3 2
    // 3 7 1 2 3 3
    hisa::hisa_cpu h(2);
    h.load_vectical(test_raw);

    h.deduplicate();
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < h.columns[i].size(); j++) {
            std::cout << h.columns[i].raw_data[j] << " ";
        }
        std::cout << std::endl;
    }
    h.build_index();

    h.print_all();
}

void test_gpu_basic() {
    hisa::multi_hisa h(2);
    thrust::host_vector<thrust::host_vector<hisa::internal_data_type>>
        test_raw_host;
    test_raw_host.push_back({{1, 6, 3, 9, 1, 2, 3, 8}});
    test_raw_host.push_back({{3, 2, 3, 9, 3, 7, 3, 1}});
    // 3 8 1 1 3 2
    // 3 7 1 2 3 3
    h.init_load_vectical(test_raw_host, 8);
    auto start = std::chrono::high_resolution_clock::now();
    h.deduplicate();
    std::cout << "deduplicate done" << std::endl;
    thrust::host_vector<hisa::internal_data_type> data_host;
    for (int i = 0; i < 2; i++) {
        data_host = h.data[i];
        for (int j = 0; j < data_host.size(); j++) {
            std::cout << data_host[j] << " ";
        }
        std::cout << std::endl;
    }
    h.persist_newt();
    std::cout << "persist_newt done" << std::endl;
    hisa::device_data_t result_key = test_raw_host[0];
    thrust::host_vector<uint32_t> result_key_host = result_key;
    hisa::device_ranges_t result_value(8);
    // auto &map_ptr = h.full_columns[0].unique_v_map;
    // map_ptr->find(result_key.begin(), result_key.end(), result_value.begin());
    h.full_columns[0].unique_v_map_simp.find(result_key, result_value);

    thrust::host_vector<hisa::comp_range_t> result_value_host = h.full_columns[0].unique_v_map_simp.values;
    for (int i = 0; i < result_value_host.size(); i++) {
        auto offset = static_cast<uint32_t>(result_value_host[i] >> 32);
        auto length = static_cast<uint32_t>(result_value_host[i]);
        std::cout << "(" << result_key_host[i] << " : " << offset << " "
                  << length << ") ";
    }
    std::cout << std::endl;
    h.print_all();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "total time: " << duration.count() << std::endl;
}

int main() {
    rmm::mr::cuda_memory_resource cuda_mr{};
    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> mr{
        &cuda_mr, 4 * 256 * 1024};
    rmm::mr::set_current_device_resource(&mr);

    signal(SIGSEGV, handler);
    // generate 3 columns of random data
    thrust::host_vector<thrust::host_vector<uint32_t>> data_columns_vertical(
        TEST_ARIRTY);
    thrust::host_vector<uint32_t> data_columns_horizontal;

    for (int i = 0; i < TEST_ARIRTY; i++) {
        data_columns_vertical[i].resize(TEST_CASE_SIZE);
        generate_random_data(data_columns_vertical[i]);
    }
    raw_vertical_to_horizontal(data_columns_vertical, data_columns_horizontal);

    std::cout << "generate_random_data done" << std::endl;
    hisa::hisa_cpu hashtrie_cpu(TEST_ARIRTY);
    hashtrie_cpu.load_vectical(data_columns_vertical);
    hashtrie_cpu.build_index();

    std::cout << "hisa_cpu >>>>>>>> " << std::endl;
    testcase_deduplicate();
    std::cout << "hisa_gpu >>>>>>>> " << std::endl;
    test_gpu_basic();

    // uint64_t total_hisa_load_time = 0;
    // double total_hisa_hash_time = 0;
    // uint64_t total_hashtrie_cpu_load_time = 0;
    // uint64_t total_hash_time = 0;
    // uint64_t total_muti_hisa_load_time = 0;
    // uint64_t total_muti_hisa_hash_time = 0;
    // uint64_t total_muti_hisa_dedup_time = 0;
    // uint64_t total_hash_time_sort_time = 0;
    // uint64_t total_muti_hisa_h2d_time = 0;

    // for (int i = 0; i < REPEAT; i++) {
    //     std::cout << "repeat: " << i << std::endl;
    //     // test load
    //     hisa::hisa_cpu hashtrie_cpu2(TEST_ARIRTY);
    //     auto start = std::chrono::high_resolution_clock::now();
    //     // test_load(data_columns_vertical, hashtrie_cpu2);
    //     auto end = std::chrono::high_resolution_clock::now();
    //     auto duration =
    //         std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    //     total_hashtrie_cpu_load_time += duration.count();
    //     total_hash_time += hashtrie_cpu2.hash_time;

    //     // test load hisa
    //     Relation *rel = new Relation();
    //     auto start_hisa = std::chrono::high_resolution_clock::now();
    //     test_load_hisa(data_columns_horizontal, rel);
    //     auto end_hisa = std::chrono::high_resolution_clock::now();
    //     auto duration_hisa =
    //         std::chrono::duration_cast<std::chrono::microseconds>(end_hisa -
    //                                                               start_hisa);
    //     total_hisa_load_time += duration_hisa.count();
    //     total_hisa_hash_time += rel->detail_time[2];
    //     rel->drop();
    //     delete rel;

    //     // test load multi
    //     hisa::multi_hisa h(TEST_ARIRTY);
    //     auto start_multi = std::chrono::high_resolution_clock::now();
    //     test_load_multi(data_columns_vertical, h);
    //     auto end_multi = std::chrono::high_resolution_clock::now();
    //     auto duration_multi =
    //         std::chrono::duration_cast<std::chrono::microseconds>(end_multi -
    //                                                               start_multi);
    //     h.clear();
    //     total_muti_hisa_load_time += duration_multi.count();
    //     total_muti_hisa_hash_time += h.hash_time;
    //     total_muti_hisa_dedup_time += h.dedup_time;
    //     total_hash_time_sort_time += h.sort_time;
    //     total_muti_hisa_h2d_time += h.load_time;
    // }

    // std::cout << "total_hashtrie_cpu_load_time: "
    //           << total_hashtrie_cpu_load_time << std::endl;
    // std::cout << "total hash time : " << total_hash_time << std::endl;
    // std::cout << "total_hisa_load_time: " << total_hisa_load_time << std::endl;
    // std::cout << "total_hisa_hash_time: " << total_hisa_hash_time << std::endl;
    // std::cout << "total_muti_hisa_load_time: " << total_muti_hisa_load_time
    //           << std::endl;
    // std::cout << "total_muti_hisa_hash_time: " << total_muti_hisa_hash_time
    //           << std::endl;
    // std::cout << "total_muti_hisa_dedup_time: " << total_muti_hisa_dedup_time
    //           << std::endl;
    // std::cout << "total_hash_time_sort_time: " << total_hash_time_sort_time
    //           << std::endl;
    // std::cout << "total_muti_hisa_h2d_time: " << total_muti_hisa_h2d_time
    //           << std::endl;

    return 0;
}

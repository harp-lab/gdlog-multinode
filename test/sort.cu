

// compare function for sort and merge on CPU and GPU

// pacakge for time measurement
#include <chrono>
#include <iostream>

// #define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_OMP
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_TBB

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

#include "../include/timer.cuh"


void sort_on_cpu(thrust::host_vector<uint32_t> &data_column1,
                 thrust::host_vector<uint32_t> &data_column2,
                 thrust::host_vector<uint32_t> &indices) {
    thrust::sequence(indices.begin(), indices.end());
    thrust::stable_sort_by_key(data_column1.begin(), data_column1.end(), indices.begin());
    thrust::host_vector<uint32_t> sorted_data_column1(data_column1.size());
    thrust::host_vector<uint32_t> sorted_data_column2(data_column2.size());
    thrust::gather(indices.begin(), indices.end(), data_column2.begin(), sorted_data_column2.begin());
    thrust::stable_sort_by_key(sorted_data_column2.begin(), sorted_data_column2.end(), indices.begin());
    data_column1 = sorted_data_column1;
    data_column2 = sorted_data_column2;
}

void merge_on_cpu(thrust::host_vector<uint32_t> &data_column1,
                  thrust::host_vector<uint32_t> &data_column2,
                  thrust::host_vector<uint32_t> &merged_data_column1) {
    thrust::merge(data_column1.begin(), data_column1.end(), data_column2.begin(), data_column2.end(), merged_data_column1.begin());
}

void merge_on_gpu(thrust::device_vector<uint32_t> &data_column1,
                  thrust::device_vector<uint32_t> &data_column2,
                  thrust::device_vector<uint32_t> &merged_data_column1) {
    thrust::merge(data_column1.begin(), data_column1.end(), data_column2.begin(), data_column2.end(), merged_data_column1.begin());
}

void sort_on_gpu(thrust::device_vector<uint32_t> &data_column1,
                 thrust::device_vector<uint32_t> &data_column2,
                 thrust::device_vector<uint32_t> &indices) {
    thrust::sequence(indices.begin(), indices.end());
    thrust::stable_sort_by_key(data_column1.begin(), data_column1.end(), indices.begin());
    thrust::device_vector<uint32_t> sorted_data_column1(data_column1.size());
    thrust::device_vector<uint32_t> sorted_data_column2(data_column2.size());
    thrust::gather(indices.begin(), indices.end(), data_column2.begin(), sorted_data_column2.begin());
    thrust::stable_sort_by_key(sorted_data_column2.begin(), sorted_data_column2.end(), indices.begin());
    data_column1 = sorted_data_column1;
    data_column2 = sorted_data_column2;
}

void generate_random_data(thrust::host_vector<uint32_t> &data_column1, thrust::host_vector<uint32_t> &data_column2) {
    for (int i = 0; i < data_column1.size(); i++) {
        data_column1[i] = rand() % (1000 * 1000);
    }
    for (int i = 0; i < data_column2.size(); i++) {
        data_column2[i] = rand() % (1000 * 1000);
    }
}

void copy_data_to_gpu(thrust::host_vector<uint32_t> &data_column1, thrust::host_vector<uint32_t> &data_column2,
                      thrust::device_vector<uint32_t> &data_column1_gpu, thrust::device_vector<uint32_t> &data_column2_gpu) {
    data_column1_gpu = data_column1;
    data_column2_gpu = data_column2;
}

int main() {

    // generate random data 1000 * 1000

    int repeat = 1000;

    thrust::host_vector<uint32_t> data_column1_raw(1000 * 1000);
    thrust::host_vector<uint32_t> data_column2_raw(1000 * 1000);
    thrust::device_vector<uint32_t> data_column1_raw_d(1000 * 1000);
    thrust::device_vector<uint32_t> data_column2_raw_d(1000 * 1000);
    generate_random_data(data_column1_raw, data_column2_raw);
    copy_data_to_gpu(data_column1_raw, data_column2_raw, data_column1_raw_d, data_column2_raw_d);
  
    thrust::host_vector<uint32_t> indices(data_column1_raw.size());
    thrust::device_vector<uint32_t> indices_gpu(data_column1_raw.size());

    auto total_sort_time_cpu = 0;
    auto total_sort_time_gpu = 0;
    auto total_merge_time_cpu = 0;
    auto total_merge_time_gpu = 0;
    auto total_mem_time_cpu = 0;
    auto total_mem_time_gpu = 0;
    for (int i = 0; i < repeat; i++) {
        std::cout << "Iteration: " << i << std::endl;
        auto begin_mem = std::chrono::high_resolution_clock::now();
        thrust::host_vector<uint32_t> data_column1(1000 * 1000);
        thrust::host_vector<uint32_t> data_column2(1000 * 1000);
        data_column1 = data_column1_raw;
        data_column2 = data_column2_raw;
        auto end_mem = std::chrono::high_resolution_clock::now();
        thrust::host_vector<uint32_t> merged_data_column_cpu(data_column1.size() * 2);
        total_mem_time_cpu += std::chrono::duration_cast<std::chrono::microseconds>(end_mem - begin_mem).count();

        begin_mem = std::chrono::high_resolution_clock::now();
        thrust::device_vector<uint32_t> data_column1_gpu(1000 * 1000);
        thrust::device_vector<uint32_t> data_column2_gpu(1000 * 1000);
        data_column1_gpu = data_column1_raw_d;
        data_column2_gpu = data_column2_raw_d;
        end_mem = std::chrono::high_resolution_clock::now();
        thrust::device_vector<uint32_t> merged_data_column_gpu(data_column1.size() * 2);
        total_mem_time_gpu += std::chrono::duration_cast<std::chrono::microseconds>(end_mem - begin_mem).count();

        auto begin = std::chrono::high_resolution_clock::now();
        sort_on_cpu(data_column1, data_column2, indices);
        auto end = std::chrono::high_resolution_clock::now();
        total_sort_time_cpu += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        begin = std::chrono::high_resolution_clock::now();
        sort_on_gpu(data_column1_gpu, data_column2_gpu, indices_gpu);
        end = std::chrono::high_resolution_clock::now();
        total_sort_time_gpu += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

        // sort column2
        thrust::sort(data_column2.begin(), data_column2.end());
        thrust::sort(data_column2_gpu.begin(), data_column2_gpu.end());

        begin = std::chrono::high_resolution_clock::now();
        merge_on_cpu(data_column1, data_column2, merged_data_column_cpu);
        end = std::chrono::high_resolution_clock::now();
        total_merge_time_cpu += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        begin = std::chrono::high_resolution_clock::now();
        merge_on_gpu(data_column1_gpu, data_column2_gpu, merged_data_column_gpu);
        end = std::chrono::high_resolution_clock::now();
        total_merge_time_gpu += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();        
    }
    std::cout << "CPU Sort time: " << total_sort_time_cpu << std::endl;
    std::cout << "GPU Sort time: " << total_sort_time_gpu << std::endl;
    std::cout << "CPU Mem time: " << total_mem_time_cpu << std::endl;
    std::cout << "GPU Mem time: " << total_mem_time_gpu << std::endl;
    std::cout << "CPU Merge time: " << total_merge_time_cpu << std::endl;
    std::cout << "GPU Merge time: " << total_merge_time_gpu << std::endl;

    return 0;
}


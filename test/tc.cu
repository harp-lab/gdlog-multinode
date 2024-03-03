#include <chrono>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <thrust/execution_policy.h>
#include <thrust/merge.h>
#include <thrust/set_operations.h>
#include <vector>

#include "../include/exception.cuh"
#include "../include/lie.cuh"
#include "../include/print.cuh"
#include "../include/timer.cuh"

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

void analysis_bench(int argc, char *argv[], int block_size, int grid_size) {
    const char *dataset_path = argv[1];
    KernelTimer timer;
    int relation_columns = 2;
    std::chrono::high_resolution_clock::time_point time_point_begin;
    std::chrono::high_resolution_clock::time_point time_point_end;
    time_point_begin = std::chrono::high_resolution_clock::now();
    double spent_time;

    // load the raw graph
    thrust::host_vector<column_type> raw_graph_data_vec;
    std::map<column_type, std::string> string_map;
    file_to_buffer(dataset_path, raw_graph_data_vec, string_map);
    tuple_size_t graph_edge_counts = raw_graph_data_vec.size() / 2;
    column_type *raw_graph_data = raw_graph_data_vec.data();
    // std::cout << "reversing graph ... " << graph_edge_counts * 2 * sizeof(column_type) << std::endl;
    column_type *raw_reverse_graph_data =
        (column_type *)malloc(graph_edge_counts * 2 * sizeof(column_type));

    for (tuple_size_t i = 0; i < graph_edge_counts; i++) {
        raw_reverse_graph_data[i * 2 + 1] = raw_graph_data[i * 2];
        raw_reverse_graph_data[i * 2] = raw_graph_data[i * 2 + 1];
    }
    // std::cout << "finish reverse graph." << std::endl;

    timer.start_timer();
    Relation *edge_2__2_1 = new Relation();
    // cudaMallocHost((void **)&edge_2__2_1, sizeof(Relation));
    Relation *path_2__1_2 = new Relation();
    path_2__1_2->index_flag = false;
    // cudaMallocHost((void **)&path_2__1_2, sizeof(Relation));
    // std::cout << "edge size " << graph_edge_counts << std::endl;
    load_relation(path_2__1_2, "path_2__1_2", 2, raw_graph_data,
                  graph_edge_counts, 1, 0, grid_size, block_size);
    load_relation(edge_2__2_1, "edge_2__2_1", 2, raw_reverse_graph_data,
                  graph_edge_counts, 1, 0, grid_size, block_size);
    timer.stop_timer();
    // double kernel_spent_time = timer.get_spent_time();
    // std::cout << "Build hash table time: " << timer.get_spent_time()
    //           << std::endl;

    timer.start_timer();
    Communicator comm;
    comm.init(argc, argv);
    LIE tc_scc(grid_size, block_size);
    tc_scc.set_communicator(&comm);

    tc_scc.reload_full_flag = false;
    tc_scc.add_relations(edge_2__2_1, true);
    tc_scc.add_relations(path_2__1_2, false);
    float join_detail[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int> join_order = {1, 3};
    TupleGenerator tp1_hook(2, 2, join_order);
    tc_scc.add_ra(RelationalJoin(edge_2__2_1, FULL, path_2__1_2, DELTA,
                                 path_2__1_2, tp1_hook,
                                 grid_size, block_size, join_detail));

    tc_scc.fixpoint_loop();
    timer.stop_timer();

    int total_tuples = path_2__1_2->full->tuple_counts;
    total_tuples = comm.reduceSumTupleSize(total_tuples);
    if (comm.getRank() == 0) {
        std::cout << "Path counts " << total_tuples << std::endl;
        std::cout << "TC time: " << timer.get_spent_time() << std::endl;
        std::cout << "join detail: " << std::endl;
        std::cout << "compute size time:  " << join_detail[0] << std::endl;
        std::cout << "reduce + scan time: " << join_detail[1] << std::endl;
        std::cout << "fetch result time:  " << join_detail[2] << std::endl;
        std::cout << "sort time:          " << join_detail[3] << std::endl;
        std::cout << "build index time:   " << join_detail[5] << std::endl;
        std::cout << "merge time:         " << join_detail[6] << std::endl;
        std::cout << "unique time:        " << join_detail[4] + join_detail[7]
                  << std::endl;
    }
}

int main(int argc, char *argv[]) {
    int device_id;
    int number_of_sm;
    cudaGetDevice(&device_id);
    cudaDeviceGetAttribute(&number_of_sm, cudaDevAttrMultiProcessorCount,
                           device_id);
    std::cout << "num of sm " << number_of_sm << std::endl;
    // std::cout << "using " << EMPTY_HASH_ENTRY << " as empty hash entry"
    //           << std::endl;
    int block_size, grid_size;
    block_size = 512;
    grid_size = 32 * number_of_sm;
    std::locale loc("");
    rmm::mr::cuda_memory_resource cuda_mr{};
    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> mr{&cuda_mr, 4 * 256 * 1024 };
    // rmm::mr::managed_memory_resource mr;
    // rmm::mr::arena_memory_resource<rmm::mr::device_memory_resource> mr{&cuda_mr};

    rmm::mr::set_current_device_resource(&mr);
    analysis_bench(argc, argv, block_size, grid_size);
    return 0;
}

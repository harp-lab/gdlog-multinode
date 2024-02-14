
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdlib.h>

#include <vector>

#include "../include/exception.cuh"
#include "../include/lie.cuh"
#include "../include/print.cuh"
#include "../include/timer.cuh"

#include "builtin.h"

column_type raw_graph_data[20] = {1, 2, 1, 5, 1, 6, 2, 3, 2, 6,
                                  3, 4, 8, 7, 4, 5, 4, 6, 5, 6};
tuple_size_t graph_edge_counts = 10;

void filter_test(int block_size, int grid_size) {
    column_type *raw_reverse_graph_data =
        (column_type *)malloc(graph_edge_counts * 2 * sizeof(column_type));

    for (tuple_size_t i = 0; i < graph_edge_counts; i++) {
        raw_reverse_graph_data[i * 2 + 1] = raw_graph_data[i * 2];
        raw_reverse_graph_data[i * 2] = raw_graph_data[i * 2 + 1];
    }
    std::cout << "finish reverse graph." << std::endl;
    Relation *edge_2__2_1 = nullptr;
    Relation *path_2__1_2 = nullptr;
    // assert(edge_2__2_1 != nullptr);
    // assert(path_2__1_2 != nullptr);
    edge_2__2_1 = new Relation();
    // cudaMallocHost((void **)&edge_2__2_1, sizeof(Relation));
    path_2__1_2 = new Relation();
    path_2__1_2->index_flag = false;
    // cudaMallocHost((void **)&path_2__1_2, sizeof(Relation));
    load_relation(path_2__1_2, "path_2__1_2", 2, raw_graph_data, graph_edge_counts, 1,
                  0, grid_size, block_size);
    load_relation(edge_2__2_1, "edge_2__2_1", 2, raw_reverse_graph_data,
                  graph_edge_counts, 1, 0, grid_size, block_size);
    // std::cout << edge_2__2_1->name << std::endl;

    // double kernel_spent_time = timer.get_spent_time();
    LIE tc_scc(grid_size, block_size);
    tc_scc.reload_full_flag = false;
    tc_scc.add_relations(edge_2__2_1, true);
    tc_scc.add_relations(path_2__1_2, false);
    float join_detail[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int> join_order = {1, 3};

    std::vector<BinaryFilterComparison> ops = {BinaryFilterComparison::EMPTY,
                                               BinaryFilterComparison::LE};
    std::vector<int> left = {0, 1};
    std::vector<int> right = {0, (-4 - 15)};
    TupleFilter tp_filter(2, ops, left, right);
    column_type t_test[2] = {1,4};
    std::cout << tp_filter(t_test) << std::endl;
    assert(tp_filter(t_test) == false);
    tc_scc.add_ra(
        RelationalFilter(edge_2__2_1, FULL, tp_filter, grid_size, block_size));

    tc_scc.fixpoint_loop();

    print_tuple_rows(edge_2__2_1->full, "edge_2__2_1");
    assert(edge_2__2_1->full->size() == 6);

    std::cout << "Filter test passed" << std::endl;
}

int main(int argc, char *argv[]) {
    int device_id;
    int number_of_sm;
    cudaGetDevice(&device_id);
    cudaDeviceGetAttribute(&number_of_sm, cudaDevAttrMultiProcessorCount,
                           device_id);
    std::cout << "num of sm " << number_of_sm << std::endl;
    std::cout << "using " << EMPTY_HASH_ENTRY << " as empty hash entry"
              << std::endl;
    int block_size, grid_size;
    block_size = 512;
    grid_size = 32 * number_of_sm;
    std::locale loc("");

    filter_test(block_size, grid_size);
    return 0;
}


#include <chrono>
#include <fstream>
#include <iostream>
#include <stdlib.h>

#include <vector>

#include "exception.cuh"
#include "lie.cuh"
#include "print.cuh"
#include "timer.cuh"

#include "builtin.h"


column_type raw_graph_data[20] = {1, 2, 1, 5, 1, 6, 2, 3, 2, 6,
                                  3, 4, 8, 7, 4, 5, 4, 6, 5, 6};
tuple_size_t graph_edge_counts = 10;

void project_test(Communicator *communicator, int block_size, int grid_size) {
    Relation *edge_2__2_1 = new Relation();
    Relation *path_2__1_2 = new Relation();

    load_relation(edge_2__2_1, "edge_2__2_1", 2, raw_graph_data,
                  graph_edge_counts, 1, 0, grid_size, block_size);
    load_relation(path_2__1_2, "path_2__1_2", 2, nullptr, 0, 1, 0, grid_size,
                  block_size);

    LIE cp_scc(grid_size, block_size);
    cp_scc.add_relations(edge_2__2_1, true);
    cp_scc.add_relations(path_2__1_2, false);
    cp_scc.set_communicator(communicator);
    cp_scc.add_ra(RelationalCopy(edge_2__2_1, FULL, path_2__1_2,
                                 TupleProjector(2, {1, 1}), grid_size,
                                 block_size));

    cp_scc.fixpoint_loop();
    // print_tuple_rows(path_2__1_2->full, "path_2__1_2");
    for (int i = 0; i < communicator->getTotalRank(); i++) {
        if (communicator->getRank() == i) {
            print_tuple_rows(path_2__1_2->full, "path_2__1_2");
            if (communicator->getTotalRank() == 1 &&
                communicator->getRank() == 0) {
                assert(path_2__1_2->full->tuple_counts == 6);
            }
            if (communicator->getTotalRank() == 2 &&
                communicator->getRank() == 0) {
                assert(path_2__1_2->full->tuple_counts == 3);
            }
        }
        communicator->barrier();
    }

    if (communicator->getRank() == 0) {
        std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Project test passed" << std::endl;
    }
}

void filter_test(Communicator *communicator, int block_size, int grid_size) {
    column_type *raw_reverse_graph_data =
        (column_type *)malloc(graph_edge_counts * 2 * sizeof(column_type));

    for (tuple_size_t i = 0; i < graph_edge_counts; i++) {
        raw_reverse_graph_data[i * 2 + 1] = raw_graph_data[i * 2];
        raw_reverse_graph_data[i * 2] = raw_graph_data[i * 2 + 1];
    }
    Relation *edge_2__2_1 = nullptr;
    Relation *path_2__1_2 = nullptr;
    // assert(edge_2__2_1 != nullptr);
    // assert(path_2__1_2 != nullptr);
    edge_2__2_1 = new Relation();
    // cudaMallocHost((void **)&edge_2__2_1, sizeof(Relation));
    path_2__1_2 = new Relation();
    path_2__1_2->index_flag = false;
    // cudaMallocHost((void **)&path_2__1_2, sizeof(Relation));
    load_relation(path_2__1_2, "path_2__1_2", 2, raw_graph_data,
                  graph_edge_counts, 1, 0, grid_size, block_size);
    load_relation(edge_2__2_1, "edge_2__2_1", 2, raw_reverse_graph_data,
                  graph_edge_counts, 1, 0, grid_size, block_size);

    // double kernel_spent_time = timer.get_spent_time();
    LIE tc_scc(grid_size, block_size);

    tc_scc.set_communicator(communicator);
    tc_scc.reload_full_flag = false;
    tc_scc.add_relations(edge_2__2_1, true);
    tc_scc.add_relations(path_2__1_2, false);

    TupleFilter tp_filter(
        2, {BinaryFilterComparison::EMPTY, BinaryFilterComparison::LE},
        {EMPTY_COLUMN, 1}, {EMPTY_COLUMN, C_NUM(4)});
    column_type t_test[2] = {1, 4};
    std::cout << tp_filter(t_test) << std::endl;
    assert(tp_filter(t_test) == false);
    tc_scc.add_ra(RelationalFilter(edge_2__2_1, FULL, tp_filter));

    tc_scc.fixpoint_loop();

    print_tuple_rows(edge_2__2_1->full, "edge_2__2_1");
    for (int i = 0; i < communicator->getTotalRank(); i++) {
        if (communicator->getRank() == i) {
            print_tuple_rows(edge_2__2_1->full, "edge_2__2_1");
            if (communicator->getTotalRank() == 1 &&
                communicator->getRank() == 0) {
                assert(edge_2__2_1->full->tuple_counts == 6);
            }
            if (communicator->getTotalRank() == 2 &&
                communicator->getRank() == 0) {
                assert(edge_2__2_1->full->tuple_counts == 2);
            }
        }
        communicator->barrier();
    }

    if (communicator->getRank() == 0) {
        std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Filter test passed" << std::endl;
    }
}

void arithm_test(Communicator *communicator, int block_size, int grid_size) {
    column_type *raw_reverse_graph_data =
        (column_type *)malloc(graph_edge_counts * 2 * sizeof(column_type));

    for (tuple_size_t i = 0; i < graph_edge_counts; i++) {
        raw_reverse_graph_data[i * 2 + 1] = raw_graph_data[i * 2];
        raw_reverse_graph_data[i * 2] = raw_graph_data[i * 2 + 1];
    }
    Relation *edge_2__2_1 = nullptr;
    Relation *path_2__1_2 = nullptr;
    // assert(edge_2__2_1 != nullptr);
    // assert(path_2__1_2 != nullptr);
    edge_2__2_1 = new Relation();
    // cudaMallocHost((void **)&edge_2__2_1, sizeof(Relation));
    path_2__1_2 = new Relation();
    path_2__1_2->index_flag = false;
    // cudaMallocHost((void **)&path_2__1_2, sizeof(Relation));
    load_relation(path_2__1_2, "path_2__1_2", 2, raw_graph_data,
                  graph_edge_counts, 1, 0, grid_size, block_size);
    load_relation(edge_2__2_1, "edge_2__2_1", 2, raw_reverse_graph_data,
                  graph_edge_counts, 1, 0, grid_size, block_size);

    // double kernel_spent_time = timer.get_spent_time();
    LIE tc_scc(grid_size, block_size);
    tc_scc.set_communicator(communicator);
    tc_scc.reload_full_flag = false;
    tc_scc.add_relations(edge_2__2_1, true);
    tc_scc.add_relations(path_2__1_2, false);

    TupleArithmetic tp_arithm(
        2, {BinaryArithmeticOperator::ADD, BinaryArithmeticOperator::EMPTY},
        {0, EMPTY_COLUMN}, {C_NUM(1), 0});

    tc_scc.add_ra(RelationalArithm(edge_2__2_1, FULL, tp_arithm));

    tc_scc.fixpoint_loop();

    for (int i = 0; i < communicator->getTotalRank(); i++) {
        if (communicator->getRank() == i) {
            print_tuple_rows(edge_2__2_1->full, "edge_2__2_1");
            if (communicator->getTotalRank() == 1 &&
                communicator->getRank() == 0) {
                assert(edge_2__2_1->full->tuple_counts == 10);
            }
            if (communicator->getTotalRank() == 2 &&
                communicator->getRank() == 0) {
                assert(edge_2__2_1->full->tuple_counts == 4);
            }
        }
        communicator->barrier();
    }

    if (communicator->getRank() == 0) {
        std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Arithm test passed" << std::endl;
    }
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
    Communicator communicator;
    communicator.init(argc, argv);
    filter_test(&communicator, block_size, grid_size);
    arithm_test(&communicator, block_size, grid_size);
    project_test(&communicator, block_size, grid_size);
    return 0;
}

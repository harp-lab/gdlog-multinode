
// a test for iterative join.
// use transitive closure


#include "comm.h"
#include "exception.cuh"
#include "relation.cuh"
#include "lie.cuh"

#include "print.cuh"
#include <iostream>
#include <cstdlib>

column_type raw_graph_data[20] = {1, 2, 1, 5, 1, 6, 2, 3, 2, 6,
                                  3, 4, 3, 7, 4, 5, 4, 6, 5, 6};
tuple_size_t graph_edge_counts = 10;

bool tc_test(int argc, char **argv) {
    Communicator comm;
    comm.init(argc, argv);
    comm.barrier();
    int device_id;
    int number_of_sm;
    checkCuda(cudaGetDevice(&device_id));
    checkCuda(cudaDeviceGetAttribute(
        &number_of_sm, cudaDevAttrMultiProcessorCount, device_id));
    std::cout << "Rank " << comm.getRank() << " out of " << comm.getTotalRank()
              << " Current device id = " << device_id
              << " number_of_sm = " << number_of_sm << std::endl;
    int block_size = 512;
    int grid_size = 32 * number_of_sm;
    std::cout << "block_size = " << block_size << " grid_size = " << grid_size
              << std::endl;

    column_type *raw_reverse_graph_data =
        (column_type *)malloc(graph_edge_counts * 2 * sizeof(column_type));

    for (tuple_size_t i = 0; i < graph_edge_counts; i++) {
        raw_reverse_graph_data[i * 2 + 1] = raw_graph_data[i * 2];
        raw_reverse_graph_data[i * 2] = raw_graph_data[i * 2 + 1];
    }
    std::cout << "finish reverse graph." << std::endl;
   
    std::cout << "Rank " << comm.getRank() << " out of " << comm.getTotalRank()
              << " ranks" << std::endl;
    Relation *edge_2__2_1 = new Relation();
    Relation *path_2__1_2 = new Relation();
    path_2__1_2->index_flag = false;
    load_relation(path_2__1_2, "path_2__1_2", 2, raw_graph_data,
                  graph_edge_counts, 1, 0, grid_size, block_size);
    load_relation(edge_2__2_1, "edge_2__2_1", 2, raw_reverse_graph_data,
                  graph_edge_counts, 1, 0, grid_size, block_size);

    // before we start, we need to distribute the data on each rank
    comm.distribute(edge_2__2_1->full);
    // comm.barrier();
    comm.distribute(path_2__1_2->full);
    edge_2__2_1->full->build_index(grid_size, block_size);
    
    // std::cout << "Before start, print full on each rank" << std::endl;
    // for (int i = 0; i < comm.getTotalRank(); i++) {
    //     if (i == comm.getRank()) {
    //         std::cout << "Rank " << comm.getRank()  << std::endl;
    //         print_tuple_rows(edge_2__2_1->full, "Full edge_2__2_1");
    //         print_tuple_rows(path_2__1_2->full, "Full path_2__1_2");
    //     }
    //     comm.barrier();
    // }

    LIE tc_scc(grid_size, block_size);
    // tc_scc.max_iteration = 0;
    // tc_scc.reload_full_flag = false;
    // 
    tc_scc.set_communicator(&comm);
    tc_scc.reload_full_flag = false;
    tc_scc.add_relations(edge_2__2_1, true);
    tc_scc.add_relations(path_2__1_2, false);
    float join_detail[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int> join_order = {1, 3};
    TupleGenerator tp1_hook(2, 2, join_order);
    tc_scc.add_ra(RelationalJoin(
        edge_2__2_1, FULL, path_2__1_2, DELTA, path_2__1_2, tp1_hook,
        grid_size, block_size, join_detail));
    tc_scc.fixpoint_loop();
    comm.barrier();
    // print full on each rank
    for (int i = 0; i < comm.getTotalRank(); i++) {
        if (i == comm.getRank()) {
            std::cout << "Rank " << comm.getRank()  << std::endl;
            print_tuple_rows(path_2__1_2->full, "Full path_2__1_2");
        }
        comm.barrier();
    }
    comm.barrier();
    bool result = true;
    
    if (!comm.isInitialized() || comm.getTotalRank() == 1) {
        if (path_2__1_2->full->tuple_counts != 18) {
            result = false;
        }
    } else {
        if (comm.getTotalRank() == 2 && comm.getRank() == 0) {
            if (path_2__1_2->full->tuple_counts != 11) {
                result = false;
            }
        } else if (comm.getTotalRank() == 2 && comm.getRank() == 1) {
            if (path_2__1_2->full->tuple_counts != 7) {
                result = false;
            }
        } else if (comm.getTotalRank() == 3 && comm.getRank() == 0) {
            if (path_2__1_2->full->tuple_counts != 1) {
                result = false;
            }
        } else if (comm.getTotalRank() == 3 && comm.getRank() == 1) {
            if (path_2__1_2->full->tuple_counts != 9) {
                result = false;
            }
        } else if (comm.getTotalRank() == 3 && comm.getRank() == 2) {
            if (path_2__1_2->full->tuple_counts != 8) {
                result = false;
            }
        }
    }

    return result;
}

int main(int argc, char **argv) {
    auto tc_res = tc_test(argc, argv);
    if (!tc_res) {
        std::exit(EXIT_FAILURE);
    }
    return 0; 
}


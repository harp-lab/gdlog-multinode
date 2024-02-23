
// test for comm.h

#include "comm.h"
#include "exception.cuh"
#include "relation.cuh"
#include "relational_algebra.cuh"
#include "print.cuh"
#include <iostream>

column_type raw_graph_data[20] = {1, 2, 1, 5, 1, 6, 2, 3, 2, 6,
                                  3, 4, 8, 7, 4, 5, 4, 6, 5, 6};
tuple_size_t graph_edge_counts = 10;

column_type rank_0_data[10] = {2, 5, 2, 6};
tuple_size_t rank_0_counts = 2;
column_type rank_1_data[10] = {1, 4, 1, 7};
tuple_size_t rank_1_counts = 2;

bool test_split_relation(int argc, char **argv) {
    Communicator comm;
    comm.init(argc, argv);
    comm.barrier();
    // test for comm.h
    int device_id;
    int number_of_sm;
    // checkCuda(cudaSetDevice(comm.getRank()));
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
   
    std::cout << "Rank " << comm.getRank() << " out of " << comm.getTotalRank()
              << " ranks" << std::endl;
    Relation *path_2__1_2 = new Relation();
    if (comm.getRank() == 0) {
        load_relation(path_2__1_2, "path_2__1_2", 2, rank_0_data,
                      rank_0_counts, 1, 0, grid_size, block_size);
    } else if (comm.getRank() == 1) {
        load_relation(path_2__1_2, "path_2__1_2", 2, rank_1_data,
                      rank_1_counts, 1, 0, grid_size, block_size);
    }
    // load_relation(path_2__1_2, "path_2__1_2", 2, raw_graph_data,
    //               graph_edge_counts, 1, 0, grid_size, block_size);
    comm.barrier();
    std::cout << "Before comm Rank " << comm.getRank() << " path_2__1_2->full ="
              << path_2__1_2->fulls[0]->tuple_counts << std::endl;
    // split full relation to all ranks
    comm.distribute_bucket(path_2__1_2->fulls[0]);
    comm.barrier();
    // std::cout << "Rank " << comm.getRank() << " path_2__1_2->full->size() ="
    //           << path_2__1_2->full->tuple_counts << std::endl;

    // let each rank sequentially print the full relation
    for (int i = 0; i < comm.getTotalRank(); i++) {
        if (i == comm.getRank()) {
            std::cout << "Rank " << comm.getRank() << " path_2__1_2->full ="
                      << std::endl;
            print_tuple_rows(path_2__1_2->fulls[0], "Full ");
        }
        comm.barrier();
    }

    return true;
}

int main(int argc, char **argv) {
    test_split_relation(argc, argv);
    return 0; 
}
